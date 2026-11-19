"""
nba_model_v2.py

A model built on the full history nba_api offers, to be judged head-to-head against the
30-game rolling model in nba_model.py on identical games.

Why the 30-game window is the weak link (grounded arithmetic, not a test):
  single-game net-rating noise is ~13 pts, so a 30-game mean has a standard error of ~2.4 pts.
  True team-strength spread is only ~4 pts. Roughly a third of the variance in a 30-game average
  is noise, it ignores who the games were played against, and it forgets everything before.

What v2 adds, using only endpoints nba_api exposes:
  1. Kalman team ratings: every game back to the first loaded season, opponent-adjusted, with
     uncertainty and a season-to-season carryover (instead of a raw 30-game mean).
  2. Multi-horizon form: exponentially weighted net rating at half-lives of 5, 15 and 45 games.
  3. Schedule load: rest, back-to-backs, games in the last 7 days.
  4. Player layer: PlayerGameLogs -> expected-minutes lineup shares -> ridge "RAPM-lite" player
     values -> lineup strength for each game. This is the only route to injury-aware predictions.

What nba_api does NOT have: injury reports. In a backtest, "who is available" is inferred from who
actually appeared (close to what an injury report says an hour before tip). For a live prediction
you must tell the model who is out:  python nba_model_v2.py compare --out "Player Name" ...
Without that, v2 assumes full health while the market knows better, and the gap is not an edge.

Usage:
    python nba_model_v2.py backtest [--seasons 8] [--retrain-days 30]
    python nba_model_v2.py compare  [--out "Name" "Name" ...] [--min-ev 0.03] [--max-edge 0.10]
"""
import argparse
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

import nba_model as v1

CACHE_DIR, OUT_DIR = v1.CACHE_DIR, v1.OUT_DIR


# =====================================================================================
# DATA: player logs (one call per season)
# =====================================================================================
def load_player_logs(n_seasons=8, refresh=True, retries=3):
    from nba_api.stats.endpoints import playergamelogs
    CACHE_DIR.mkdir(exist_ok=True)
    cur = v1.season_start_year()
    frames, keep = [], ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "MIN"]
    for y in range(cur - n_seasons + 1, cur + 1):
        s = v1.season_str(y)
        f = CACHE_DIR / f"playerlog_{s}.csv"
        if f.exists() and y < cur and len(pd.read_csv(f, usecols=["GAME_ID"])) >= 20000:
            frames.append(pd.read_csv(f, dtype={"GAME_ID": str}))
            continue
        if f.exists() and y == cur and not refresh:
            frames.append(pd.read_csv(f, dtype={"GAME_ID": str}))
            continue
        for attempt in range(retries):
            try:
                df = playergamelogs.PlayerGameLogs(
                    season_nullable=s, season_type_nullable="Regular Season", timeout=90).get_data_frames()[0]
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        if len(df):
            df = df[keep]
            df.to_csv(f, index=False)
            frames.append(df)
        time.sleep(1)
    pl = pd.concat(frames, ignore_index=True)
    pl["GAME_ID"] = pl["GAME_ID"].astype(str)
    pl["GAME_DATE"] = pd.to_datetime(pl["GAME_DATE"]).dt.tz_localize(None)
    return pl[pl["MIN"] > 0]


# =====================================================================================
# FEATURES 1-3: Kalman ratings, multi-horizon form, schedule load
# =====================================================================================
def add_kalman(games, q=0.08, obs_var=169.0, hfa=2.5, carry=0.7, p0=20.0, p_off=12.0):
    """
    Sequential opponent-adjusted team ratings on per-100-possession net rating.
      obs = (r_home - r_away) + hfa + noise(obs_var)
      between games:   rating variance grows by q          (true strength drifts)
      between seasons: rating *= carry, variance += p_off  (roster turnover, regression to mean)
    Uses a diagonal approximation (home/away posteriors treated as independent after each update).
    Outputs are PRE-game: kr_diff (home minus away rating) and kr_unc (combined std dev).
    """
    n = len(games)
    diff, unc = np.full(n, np.nan), np.full(n, np.nan)
    r, P, last = {}, {}, None
    for i, (h, a, dt, y) in enumerate(zip(games.h_TEAM_ID, games.a_TEAM_ID, games.date, games.h_net)):
        s = v1.season_start_year(dt)
        if last is not None and s != last:
            for t in r:
                r[t] *= carry
                P[t] = P[t] * carry ** 2 + p_off
        last = s
        rh, ra, Ph, Pa = r.get(h, 0.0), r.get(a, 0.0), P.get(h, p0), P.get(a, p0)
        diff[i], unc[i] = rh - ra, np.sqrt(Ph + Pa)
        if not np.isnan(y):
            S = Ph + Pa + obs_var
            v = y - (rh - ra + hfa)
            rh, ra = rh + Ph / S * v, ra - Pa / S * v
            Ph, Pa = Ph - Ph * Ph / S + q, Pa - Pa * Pa / S + q
        r[h], r[a], P[h], P[a] = rh, ra, Ph, Pa
    out = games.copy()
    out["kr_diff"], out["kr_unc"] = diff, unc
    return out


def add_team_form(tl):
    tl = tl.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    g = tl.groupby("TEAM_ID", sort=False)
    for h in (5, 15, 45):
        tl[f"ew{h}"] = g["net_rtg"].transform(lambda s: s.shift(1).ewm(halflife=h, min_periods=5).mean())
    n7 = np.zeros(len(tl))
    dates = tl["GAME_DATE"].to_numpy()
    for _, idx in g.indices.items():
        d = dates[idx]
        n7[idx] = np.arange(len(idx)) - np.searchsorted(d, d - np.timedelta64(7, "D"), side="left")
    tl["n7"] = n7  # games played in the 7 days before this one
    return tl


def _attach_side(games, tl, cols):
    sub = tl[["GAME_ID", "is_home"] + cols]
    for side, flag in (("h", 1), ("a", 0)):
        s = sub[sub.is_home == flag].drop(columns="is_home").rename(columns={c: f"{side}_{c}" for c in cols})
        games = games.merge(s, on="GAME_ID", how="left")
    return games


# =====================================================================================
# FEATURE 4: player layer (expected-minutes lineup strength from ridge player values)
# =====================================================================================
def _live_shares(pl, team_id, out_ids, top=10):
    """Expected minute shares for an upcoming game: recent rotation minus anyone ruled out."""
    t = pl[pl.TEAM_ID == team_id]
    recent = t.sort_values("GAME_DATE").GAME_ID.drop_duplicates().tail(5)
    r = t[t.GAME_ID.isin(recent)].sort_values("GAME_DATE").groupby("PLAYER_ID").tail(1)
    r = r[~r.PLAYER_ID.isin(out_ids)].nlargest(top, "exp_incl")
    return (r.set_index("PLAYER_ID").exp_incl / r.exp_incl.sum()).to_dict()


def build_lineup_matrix(pl, games, out_names=()):
    """
    Sparse design matrix, one row per game: +share for each home player, -share for each away player.
    Shares come from PRE-game expected minutes (prior 10 appearances), restricted to players who
    appeared. In-game minutes are never used, because they depend on the result (blowouts).
    """
    pl = pl.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    m = pl.groupby("PLAYER_ID", sort=False)["MIN"]
    pl["exp_pre"] = m.transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean()).fillna(10.0)
    pl["exp_incl"] = m.transform(lambda s: s.rolling(10, min_periods=3).mean()).fillna(pl["MIN"])
    pids = {p: i for i, p in enumerate(pl.PLAYER_ID.unique())}

    home = games.set_index("GAME_ID").h_TEAM_ID
    hist = pl[pl.GAME_ID.isin(home.index)].copy()
    hist["share"] = hist.exp_pre / hist.groupby(["GAME_ID", "TEAM_ID"]).exp_pre.transform("sum")
    hist["sign"] = np.where(hist.TEAM_ID == hist.GAME_ID.map(home), 1.0, -1.0)
    gids = list(hist.GAME_ID.unique())
    gid_idx = {g: i for i, g in enumerate(gids)}
    rows = hist.GAME_ID.map(gid_idx).to_numpy()
    cols = hist.PLAYER_ID.map(pids).to_numpy()
    data = (hist.share * hist.sign).to_numpy()

    # scheduled games: expected rotation minus the players the user says are out
    out_ids = set(pl[pl.PLAYER_NAME.str.lower().isin([n.lower() for n in out_names])].PLAYER_ID)
    missing = [n for n in out_names if n.lower() not in set(pl.PLAYER_NAME.str.lower())]
    if missing:
        print("WARNING: no player match for:", missing)
    sched = games[games.GAME_ID.str.startswith("sched_")]
    r2, c2, d2 = [], [], []
    for gid, h, a in zip(sched.GAME_ID, sched.h_TEAM_ID, sched.a_TEAM_ID):
        gid_idx[gid] = len(gid_idx)
        for team, sign in ((h, 1.0), (a, -1.0)):
            for pid, sh in _live_shares(pl, team, out_ids).items():
                r2.append(gid_idx[gid]); c2.append(pids[pid]); d2.append(sign * sh)
    X = sparse.coo_matrix((np.r_[data, d2], (np.r_[rows, r2], np.r_[cols, c2])),
                          shape=(len(gid_idx), len(pids))).tocsr()
    return {"X": X, "gid_idx": gid_idx, "alpha": 5.0}


def _lineup_oof(X, y, w, alpha, folds=4):
    """Out-of-fold lineup strength for training rows, so stage 2 isn't trained on in-sample fits."""
    n = X.shape[0]
    edges = np.linspace(0, n, folds + 1).astype(int)
    oof = np.zeros(n)
    for k in range(folds):
        te = np.arange(edges[k], edges[k + 1])
        tr = np.setdiff1d(np.arange(n), te)
        oof[te] = X[te] @ Ridge(alpha=alpha).fit(X[tr], y[tr], sample_weight=w[tr]).coef_
    return oof


FEATS_V2 = ["kr_diff", "kr_unc", "d_ew5", "d_ew15", "d_ew45", "d_rest", "h_b2b", "a_b2b", "d_n7", "elo_diff"]


def _lineup_model():
    def fit(train, ctx):
        R = ctx["rapm"]
        tr = train[train.GAME_ID.isin(R["gid_idx"]) & train.h_net.notna()].copy()
        X = R["X"][tr.GAME_ID.map(R["gid_idx"]).to_numpy()]
        y = tr.h_net.to_numpy()
        w = 0.5 ** ((tr.date.max() - tr.date).dt.days.to_numpy() / 730.0)   # 2-year half-life
        coef = Ridge(alpha=R["alpha"]).fit(X, y, sample_weight=w).coef_
        tr["d_lineup"] = _lineup_oof(X, y, w, R["alpha"])
        tr = tr.dropna(subset=FEATS_V2 + ["d_lineup", "y"])
        return {"coef": coef, "m": v1._logit().fit(tr[FEATS_V2 + ["d_lineup"]], tr["y"])}

    def predict(b, test, ctx):
        R = ctx["rapm"]
        idx = test.GAME_ID.map(R["gid_idx"])
        p = np.full(len(test), np.nan)
        ok = idx.notna().to_numpy()
        if not ok.any():
            return p
        t = test[ok].copy()
        t["d_lineup"] = R["X"][idx[ok].astype(int).to_numpy()] @ b["coef"]
        X = t[FEATS_V2 + ["d_lineup"]]
        good = X.notna().all(axis=1).to_numpy()
        sub = np.full(len(t), np.nan)
        if good.any():
            sub[good] = b["m"].predict_proba(X[good])[:, 1]
        p[ok] = sub
        return p
    return fit, predict


V2_MODELS = {
    "kalman":    v1._sk_model(v1._logit, ["kr_diff"]),
    "v2_logit":  v1._sk_model(v1._logit, FEATS_V2),
    "v2_gbm":    v1._sk_model(v1._gbm, FEATS_V2),
    "v2_lineup": _lineup_model(),
}
V2_ENS = ["v2_logit", "v2_gbm", "v2_lineup"]
v1.MODELS.update(V2_MODELS)


# =====================================================================================
# PIPELINE
# =====================================================================================
def prepare_v2(n_seasons=8, refresh=True, extra_scheduled=None, out_names=()):
    games, ctx = v1.prepare(n_seasons, refresh, extra_scheduled)
    tl = add_team_form(ctx["team_log"])
    ctx["team_log"] = tl
    games = _attach_side(games, tl, ["ew5", "ew15", "ew45", "n7"])
    for h in (5, 15, 45):
        games[f"d_ew{h}"] = games[f"h_ew{h}"] - games[f"a_ew{h}"]
    games["d_n7"] = games["h_n7"] - games["a_n7"]
    home_net = tl[tl.is_home == 1].drop_duplicates("GAME_ID").set_index("GAME_ID")["net_rtg"]
    games["h_net"] = games.GAME_ID.map(home_net)
    games = add_kalman(games.sort_values(["date", "GAME_ID"]).reset_index(drop=True))
    ctx["rapm"] = build_lineup_matrix(load_player_logs(n_seasons, refresh), games, out_names)
    return games, ctx


def _game_ll(y, p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def backtest_cli(n_seasons=8, retrain_days=30):
    games, ctx = prepare_v2(n_seasons)
    models = ("home_rate",) + tuple(v1.ENSEMBLE) + tuple(V2_MODELS)
    wide, _ = v1.run_backtests(games, ctx, models, retrain_days=retrain_days)
    wide = wide.rename(columns={"ensemble": "ens_v1"})
    wide["ens_v2"] = wide[V2_ENS].mean(axis=1)
    cols = [c for c in wide.columns if c not in ("GAME_ID", "date", "y")]
    metrics = pd.DataFrame({c: v1.score(wide.y, wide[c].to_numpy()) for c in cols}).T.sort_values("log_loss")

    y = wide.y.to_numpy()
    gain = _game_ll(y, wide.ens_v1.to_numpy()) - _game_ll(y, wide.ens_v2.to_numpy())
    rng = np.random.default_rng(0)
    boots = np.array([gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(3000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    OUT_DIR.mkdir(exist_ok=True)
    wide.to_csv(OUT_DIR / "backtest_v2_predictions.csv", index=False)
    print("\n", metrics.round(4).to_string())
    print(f"\nv2 ensemble vs v1 ensemble, same {len(wide)} games: log-loss gain {gain.mean():+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    print("verdict:", "v2 is better than v1" if lo > 0 else
          "v2 is worse than v1" if hi < 0 else "no demonstrated difference between v1 and v2")
    print("saved:", v1.plot_backtests(wide, metrics, OUT_DIR / "backtest_v2.png"))


def compare_cli(out_names=(), min_ev=0.03, max_edge=0.10, n_seasons=8):
    odds = v1.load_upcoming_odds()
    if odds.empty:
        raise SystemExit("No upcoming moneylines in the DB. Run moneyline_scraper.py first.")
    if not out_names:
        print("NOTE: no --out players given; v2 assumes every rotation player is healthy.\n")
    games, ctx = prepare_v2(n_seasons, True, odds, out_names)
    train = games[games.y.notna()]
    upcoming = games[games.GAME_ID.str.startswith("sched_")]
    probs = v1.predict_probs(upcoming, ctx, train, model_names=V2_ENS)
    res = v1.compare_to_market(odds, probs, min_ev, max_edge)
    OUT_DIR.mkdir(exist_ok=True)
    f = OUT_DIR / f"compare_v2_{v1.datetime.now(v1.ET):%Y-%m-%d}.csv"
    res.to_csv(f, index=False)
    print(res.round(3).to_string(index=False))
    print("\nsaved:", f)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backtest")
    b.add_argument("--seasons", type=int, default=8)
    b.add_argument("--retrain-days", type=int, default=30)
    c = sub.add_parser("compare")
    c.add_argument("--out", nargs="*", default=[], help="players ruled out tonight, e.g. --out 'Jayson Tatum'")
    c.add_argument("--min-ev", type=float, default=0.03)
    c.add_argument("--max-edge", type=float, default=0.10)
    c.add_argument("--seasons", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "backtest":
        backtest_cli(a.seasons, a.retrain_days)
    else:
        compare_cli(a.out, a.min_ev, a.max_edge, a.seasons)


if __name__ == "__main__":
    main()