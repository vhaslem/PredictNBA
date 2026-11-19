"""
nba_model.py

1. Data + features + models that output P(home win)
2. Walk-forward backtesting and graphs
3. main(): compare today's stored moneylines against model probabilities

Usage:
    python nba_model.py backtest [--seasons 4] [--retrain-days 30]
    python nba_model.py compare  [--min-ev 0.03] [--max-edge 0.10]

Deps: pip install nba_api pandas numpy scipy scikit-learn matplotlib hmmlearn
"""
import argparse
import sqlite3
import time
import warnings
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from nba_api.stats.static import teams as nba_teams
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from moneyline_scraper import DB_PATH, init_db

warnings.filterwarnings("ignore", category=FutureWarning)
WINDOW = 30          # rolling lag, in games
MIN_PERIODS = 10     # minimum prior games before a rolling feature exists
CACHE_DIR, OUT_DIR = Path("cache"), Path("output")
ET = ZoneInfo("America/New_York")
ABBR2ID = {t["abbreviation"]: t["id"] for t in nba_teams.get_teams()}


# =====================================================================================
# 1. DATA
# =====================================================================================
def season_start_year(d=None):
    d = d or datetime.now(ET).date()
    return d.year if d.month >= 10 else d.year - 1


def season_str(y):
    return f"{y}-{(y + 1) % 100:02d}"


def load_game_logs(n_seasons=4, refresh=True, retries=3):
    """Team game logs for the last n seasons. Current season is re-pulled each call (refresh=True)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cur = season_start_year()
    frames = []
    for y in range(cur - n_seasons + 1, cur + 1):
        s = season_str(y)
        f = CACHE_DIR / f"gamelog_{s}.csv"
        complete = f.exists() and (y < cur and len(pd.read_csv(f, usecols=["GAME_ID"])) >= 2400)
        if f.exists() and (complete or (y == cur and not refresh)):
            frames.append(pd.read_csv(f, dtype={"GAME_ID": str}))
            continue
        for attempt in range(retries):
            try:
                df = leaguegamelog.LeagueGameLog(
                    season=s, player_or_team_abbreviation="T",
                    season_type_all_star="Regular Season", timeout=60).get_data_frames()[0]
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        if len(df):
            df.to_csv(f, index=False)
            frames.append(df)
        time.sleep(1)
    return pd.concat(frames, ignore_index=True)


def build_team_log(raw):
    """One row per team-game with possession-based efficiency stats."""
    d = raw.copy()
    d["GAME_ID"] = d["GAME_ID"].astype(str)
    d["GAME_DATE"] = pd.to_datetime(d["GAME_DATE"])
    d["is_home"] = d["MATCHUP"].str.contains("vs.", regex=False).astype(int)
    d["won"] = (d["WL"] == "W").astype(float)
    opp = d[["GAME_ID", "TEAM_ID", "PTS", "FGA", "FTA", "OREB", "TOV"]].rename(
        columns=lambda c: c if c == "GAME_ID" else "opp_" + c)
    d = d.merge(opp, on="GAME_ID")
    d = d[d["TEAM_ID"] != d["opp_TEAM_ID"]].copy()
    poss = d.FGA + 0.44 * d.FTA - d.OREB + d.TOV
    opp_poss = d.opp_FGA + 0.44 * d.opp_FTA - d.opp_OREB + d.opp_TOV
    d["poss"] = (poss + opp_poss) / 2
    d["off_rtg"] = 100 * d.PTS / d.poss
    d["def_rtg"] = 100 * d.opp_PTS / d.poss
    d["net_rtg"] = d.off_rtg - d.def_rtg
    d["pace"] = d.poss
    d["efg"] = (d.FGM + 0.5 * d.FG3M) / d.FGA
    d["tov_rate"] = d.TOV / d.poss
    d["margin"] = d.PTS - d.opp_PTS
    keep = ["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "is_home", "won", "margin",
            "off_rtg", "def_rtg", "net_rtg", "pace", "efg", "tov_rate"]
    return d[keep].reset_index(drop=True)


ROLL_COLS = ["net_rtg", "off_rtg", "def_rtg", "pace", "efg", "tov_rate", "won", "margin"]


def add_rolling(tl, window=WINDOW):
    """
    Rolling mean over the previous `window` games (shift(1) => the current game is never
    included, so there is no look-ahead). Continues across season boundaries.
    """
    tl = tl.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    g = tl.groupby("TEAM_ID", sort=False)
    for c in ROLL_COLS:
        tl[f"r_{c}"] = g[c].transform(lambda s: s.shift(1).rolling(window, min_periods=MIN_PERIODS).mean())
    tl["rest"] = (tl["GAME_DATE"] - g["GAME_DATE"].shift(1)).dt.days.clip(upper=7).fillna(7)
    tl["b2b"] = (tl["rest"] <= 1).astype(int)
    return tl


def build_games(tl):
    """One row per game: home/away features, their differences, and target y (home win)."""
    cols = ["GAME_ID", "GAME_DATE", "TEAM_ID", "won", "margin", "rest", "b2b"] + [f"r_{c}" for c in ROLL_COLS]
    h = tl[tl.is_home == 1][cols].add_prefix("h_")
    a = tl[tl.is_home == 0][cols].add_prefix("a_")
    g = h.merge(a, left_on="h_GAME_ID", right_on="a_GAME_ID")
    g = g.rename(columns={"h_GAME_ID": "GAME_ID", "h_GAME_DATE": "date"}).drop(columns=["a_GAME_ID", "a_GAME_DATE"])
    g["y"] = g["h_won"]
    g["margin"] = g["h_margin"]
    for c in ["net_rtg", "off_rtg", "def_rtg", "pace", "efg", "tov_rate", "won"]:
        g[f"d_{c}"] = g[f"h_r_{c}"] - g[f"a_r_{c}"]
    g["d_rest"] = g["h_rest"] - g["a_rest"]
    return g.sort_values(["date", "GAME_ID"]).reset_index(drop=True)


def add_elo(games, k=20.0, hfa=70.0, revert=0.25, base=1500.0):
    """Margin-of-victory Elo, updated sequentially. elo_diff is the pre-game rating gap (causal)."""
    R, last_season, eh, ea = {}, None, [], []
    for r in games.itertuples():
        s = season_start_year(r.date)
        if last_season is not None and s != last_season:
            R = {t: v * (1 - revert) + base * revert for t, v in R.items()}
        last_season = s
        rh, ra = R.get(r.h_TEAM_ID, base), R.get(r.a_TEAM_ID, base)
        eh.append(rh)
        ea.append(ra)
        if not np.isnan(r.margin):
            exp_h = 1 / (1 + 10 ** (-(rh + hfa - ra) / 400))
            res = 1.0 if r.margin > 0 else 0.0
            gap_w = (rh + hfa - ra) if res == 1.0 else (ra - rh - hfa)
            mult = ((abs(r.margin) + 3) ** 0.8) / (7.5 + 0.006 * gap_w)
            delta = k * mult * (res - exp_h)
            R[r.h_TEAM_ID], R[r.a_TEAM_ID] = rh + delta, ra - delta
    games = games.copy()
    games["elo_h"], games["elo_a"] = eh, ea
    games["elo_diff"] = games["elo_h"] - games["elo_a"]
    return games


def prepare(n_seasons=4, refresh=True, extra_scheduled=None):
    """
    Raw logs -> (games, ctx). `extra_scheduled` is a DataFrame of upcoming games
    [game_date, home_team, away_team] appended as outcome-less rows, so their features come
    from the exact same code path as historical games.
    """
    tl = build_team_log(load_game_logs(n_seasons, refresh))
    if extra_scheduled is not None and len(extra_scheduled):
        rows = []
        played = set(zip(tl.GAME_DATE, tl.TEAM_ID))  # skip games already in the results table
        for r in extra_scheduled.itertuples():
            if (pd.Timestamp(r.game_date), ABBR2ID[r.home_team]) in played:
                continue
            gid = f"sched_{r.game_date}_{r.away_team}_{r.home_team}"
            for ab, home in ((r.home_team, 1), (r.away_team, 0)):
                rows.append({"GAME_ID": gid, "GAME_DATE": pd.Timestamp(r.game_date),
                             "TEAM_ID": ABBR2ID[ab], "TEAM_ABBREVIATION": ab, "is_home": home})
        tl = pd.concat([tl, pd.DataFrame(rows)], ignore_index=True)
    tl = add_rolling(tl)
    games = add_elo(build_games(tl))
    return games, {"team_log": tl, "window": WINDOW}


# =====================================================================================
# 2. MODELS  (each: fit(train_games, ctx) -> bundle ; predict(bundle, games, ctx) -> P(home win))
# =====================================================================================
FEATS_ROLL = ["d_net_rtg", "d_off_rtg", "d_def_rtg", "d_efg", "d_tov_rate", "d_won", "d_pace",
              "d_rest", "h_b2b", "a_b2b"]
FEATS_HMM = ["d_hmm_em", "d_hmm_form", "d_rest", "h_b2b", "a_b2b"]
FEATS_HMM_PLUS = FEATS_HMM + ["d_net_rtg", "d_efg", "d_tov_rate", "d_won"]


def _logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))


def _gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.04, max_iter=150,
                                          min_samples_leaf=40, l2_regularization=5.0, random_state=0)


def _sk_model(make_est, feats):
    def fit(train, ctx):
        tr = train.dropna(subset=feats + ["y"])
        return {"m": make_est().fit(tr[feats], tr["y"])}

    def predict(b, test, ctx):
        X = test[feats]
        ok = X.notna().all(axis=1).to_numpy()
        p = np.full(len(test), np.nan)
        if ok.any():
            p[ok] = b["m"].predict_proba(X[ok])[:, 1]
        return p
    return fit, predict


def _home_rate():
    def fit(train, ctx):
        return {"p": train["y"].dropna().mean()}

    def predict(b, test, ctx):
        return np.full(len(test), b["p"])
    return fit, predict


# ---- Hidden Markov Model: latent "form" regime per team, Gaussian emissions on game margin
def fit_hmm_emissions(tl_train, n_states=3, restarts=3, seed=0):
    """Fit one shared K-state Gaussian HMM across all teams' margin sequences. Returns plain arrays."""
    from hmmlearn.hmm import GaussianHMM
    seqs = [g["margin"].dropna().to_numpy() for _, g in tl_train.sort_values("GAME_DATE").groupby("TEAM_ID")]
    seqs = [s for s in seqs if len(s) > 20]
    X, lengths = np.concatenate(seqs)[:, None], [len(s) for s in seqs]
    best = None
    for i in range(restarts):
        m = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, tol=1e-3,
                        min_covar=1.0, random_state=seed + i).fit(X, lengths)
        sc = m.score(X, lengths)
        if best is None or sc > best[0]:
            best = (sc, m)
    m = best[1]
    o = np.argsort(m.means_[:, 0])  # order states cold -> hot
    return {"start": m.startprob_[o], "trans": m.transmat_[np.ix_(o, o)], "means": m.means_[o, 0],
            "sds": np.sqrt(np.asarray(m.covars_).reshape(n_states, -1)[:, 0])}


def hmm_filter(params, tl, window=WINDOW):
    """
    Causal forward filter. For every team-game, restart from the prior and filter over the
    previous `window` margins only, then step the transition matrix once to get the state
    distribution going INTO that game. (hmmlearn's predict_proba is smoothed and would leak.)
    Vectorised over all rows by iterating the window offset.
    """
    tl = tl.sort_values(["TEAM_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    K, N = len(params["means"]), len(tl)
    m = tl["margin"].to_numpy(float)
    B = norm.pdf(m[:, None], params["means"][None, :], params["sds"][None, :]) + 1e-12
    B[np.isnan(m)] = 1.0  # unplayed / missing game: no information
    pos = tl.groupby("TEAM_ID").cumcount().to_numpy()
    ar = np.arange(N)
    pred = np.tile(params["start"], (N, 1))
    for k in range(window, 0, -1):
        valid = pos >= k
        Bk = np.ones((N, K))
        Bk[valid] = B[ar[valid] - k]
        a = pred * Bk
        a /= a.sum(axis=1, keepdims=True)
        pred = np.where(valid[:, None], a @ params["trans"], pred)
    return pd.DataFrame({"TEAM_ID": tl["TEAM_ID"], "GAME_ID": tl["GAME_ID"],
                         "hmm_em": pred @ params["means"],       # expected margin given regime
                         "hmm_hot": pred[:, -1], "hmm_cold": pred[:, 0]})


def get_hmm_bundle(train, ctx):
    """Fit HMM on margins up to the end of `train` only; cache per cutoff date."""
    key = train["date"].max()
    cache = ctx.setdefault("_hmm", {})
    if key not in cache:
        tl = ctx["team_log"]
        tr = tl[(tl.GAME_DATE <= key) & tl.margin.notna()]
        params = fit_hmm_emissions(tr, ctx.get("hmm_states", 3))
        cache[key] = {"params": params, "feats": hmm_filter(params, tl, ctx.get("window", WINDOW))}
    return cache[key]


def _with_hmm(df, hf):
    out = df.reset_index(drop=True)
    for s in ("h", "a"):
        f = hf.rename(columns={"TEAM_ID": f"{s}_TEAM_ID", "hmm_em": f"{s}_hmm_em",
                               "hmm_hot": f"{s}_hmm_hot", "hmm_cold": f"{s}_hmm_cold"})
        out = out.merge(f, on=["GAME_ID", f"{s}_TEAM_ID"], how="left")
    out["d_hmm_em"] = out.h_hmm_em - out.a_hmm_em
    out["d_hmm_form"] = (out.h_hmm_hot - out.h_hmm_cold) - (out.a_hmm_hot - out.a_hmm_cold)
    return out


def _hmm_model(feats):
    def fit(train, ctx):
        hb = get_hmm_bundle(train, ctx)
        tr = _with_hmm(train, hb["feats"]).dropna(subset=feats + ["y"])
        return {"hb": hb, "m": _logit().fit(tr[feats], tr["y"])}

    def predict(b, test, ctx):
        X = _with_hmm(test, b["hb"]["feats"])[feats]
        ok = X.notna().all(axis=1).to_numpy()
        p = np.full(len(test), np.nan)
        if ok.any():
            p[ok] = b["m"].predict_proba(X[ok])[:, 1]
        return p
    return fit, predict


MODELS = {
    "home_rate":  _home_rate(),                           # benchmark: always the base rate
    "elo":        _sk_model(_logit, ["elo_diff"]),
    "logit_roll": _sk_model(_logit, FEATS_ROLL),          # 30-game rolling efficiency diffs
    "gbm_roll":   _sk_model(_gbm, FEATS_ROLL),
    "hmm":        _hmm_model(FEATS_HMM),                  # HMM regime features only
    "hmm_plus":   _hmm_model(FEATS_HMM_PLUS),             # HMM + a few rolling stats
}
ENSEMBLE = ["elo", "logit_roll", "gbm_roll", "hmm_plus"]


def predict_probs(games, ctx, train, model_names=ENSEMBLE, clip=(0.02, 0.98)):
    """Fit each model on `train`, predict `games`, return per-model columns + equal-weight 'ensemble'."""
    out = pd.DataFrame({"GAME_ID": games["GAME_ID"].to_numpy()})
    for name in model_names:
        fit, predict = MODELS[name]
        out[name] = predict(fit(train, ctx), games, ctx)
    out["ensemble"] = out[list(model_names)].mean(axis=1, skipna=True)
    out[list(model_names) + ["ensemble"]] = out[list(model_names) + ["ensemble"]].clip(*clip)
    return out


# =====================================================================================
# 3. BACKTEST
# =====================================================================================
def walk_forward(games, ctx, model_name, test_start, retrain_days=30, min_train=800):
    """Expanding-window walk-forward: refit every `retrain_days`, predict the next block only."""
    fit, predict = MODELS[model_name]
    done = games[games.y.notna()].sort_values("date").reset_index(drop=True)
    edges = list(pd.date_range(pd.Timestamp(test_start), done.date.max(), freq=f"{retrain_days}D"))
    edges.append(done.date.max() + pd.Timedelta(days=1))
    parts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        train, test = done[done.date < lo], done[(done.date >= lo) & (done.date < hi)]
        if len(test) == 0 or train.dropna(subset=["d_net_rtg"]).shape[0] < min_train:
            continue
        p = predict(fit(train, ctx), test, ctx)
        parts.append(pd.DataFrame({"GAME_ID": test.GAME_ID.to_numpy(), "date": test.date.to_numpy(),
                                   "y": test.y.to_numpy(), "p": p}))
    return pd.concat(parts, ignore_index=True)


def score(y, p):
    ok = ~np.isnan(p)
    y, p = np.asarray(y)[ok], np.clip(np.asarray(p)[ok], 1e-4, 1 - 1e-4)
    return {"n": int(ok.sum()), "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p),
            "accuracy": float(((p > 0.5) == y).mean()), "auc": roc_auc_score(y, p)}


def run_backtests(games, ctx, models=tuple(MODELS), test_start=None, retrain_days=30):
    """Returns (wide predictions on the common game set, metrics table)."""
    test_start = test_start or f"{season_start_year() - 1}-10-01"
    wide = None
    for name in models:
        t0 = time.time()
        r = walk_forward(games, ctx, name, test_start, retrain_days).rename(columns={"p": name})
        wide = r if wide is None else wide.merge(r[["GAME_ID", name]], on="GAME_ID", how="outer")
        print(f"  {name:<11} {len(r):>5} games  {time.time() - t0:5.1f}s")
    comps = [m for m in ENSEMBLE if m in wide.columns]
    if len(comps) > 1:
        wide["ensemble"] = wide[comps].mean(axis=1)
    wide = wide.dropna().sort_values("date").reset_index(drop=True)  # identical games for every model
    cols = [c for c in wide.columns if c not in ("GAME_ID", "date", "y")]
    metrics = pd.DataFrame({c: score(wide.y, wide[c].to_numpy()) for c in cols}).T
    return wide, metrics.sort_values("log_loss")


def plot_backtests(wide, metrics, path=OUT_DIR / "backtest.png"):
    OUT_DIR.mkdir(exist_ok=True)
    models = [c for c in wide.columns if c not in ("GAME_ID", "date", "y")]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # (a) log loss bars
    m = metrics.sort_values("log_loss")
    ax[0, 0].barh(m.index, m.log_loss, color="#4C72B0")
    ax[0, 0].axvline(metrics.loc["home_rate", "log_loss"], color="k", ls="--", lw=1, label="home-rate baseline")
    ax[0, 0].set_xlim(m.log_loss.min() - 0.01, m.log_loss.max() + 0.01)
    ax[0, 0].set_title("Log loss (lower is better)")
    ax[0, 0].legend()

    # (b) calibration
    ax[0, 1].plot([0, 1], [0, 1], "k--", lw=1)
    for c in models:
        if c == "home_rate":
            continue
        b = pd.qcut(wide[c], 8, duplicates="drop")
        g = wide.groupby(b, observed=True).agg(p=(c, "mean"), y=("y", "mean"))
        ax[0, 1].plot(g.p, g.y, marker="o", label=c)
    ax[0, 1].set_title("Calibration (predicted vs actual home win rate)")
    ax[0, 1].set_xlabel("predicted")
    ax[0, 1].set_ylabel("actual")
    ax[0, 1].legend()

    # (c) cumulative log-loss advantage vs baseline
    def ll(c):
        p = np.clip(wide[c], 1e-4, 1 - 1e-4)
        return -(wide.y * np.log(p) + (1 - wide.y) * np.log(1 - p))
    base = ll("home_rate")
    for c in models:
        if c != "home_rate":
            ax[1, 0].plot(wide.date, (base - ll(c)).cumsum(), label=c)
    ax[1, 0].axhline(0, color="k", lw=1)
    ax[1, 0].set_title("Cumulative log-loss gain vs home-rate baseline (up = better)")
    ax[1, 0].legend()

    # (d) rolling Brier
    for c in models:
        ax[1, 1].plot(wide.date, ((wide[c] - wide.y) ** 2).rolling(200).mean(), label=c)
    ax[1, 1].set_title("Rolling 200-game Brier score (lower is better)")
    ax[1, 1].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def backtest_cli(n_seasons=4, retrain_days=30, models=None):
    games, ctx = prepare(n_seasons)
    models = tuple(dict.fromkeys(["home_rate", *(models or MODELS)]))  # baseline is required by the plots
    wide, metrics = run_backtests(games, ctx, models, retrain_days=retrain_days)
    OUT_DIR.mkdir(exist_ok=True)
    wide.to_csv(OUT_DIR / "backtest_predictions.csv", index=False)
    print("\n", metrics.round(4).to_string())
    print("\nsaved:", plot_backtests(wide, metrics))


# =====================================================================================
# 4. DAILY COMPARISON AGAINST THE MARKET
# =====================================================================================
def american_to_prob(ml):
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def american_to_decimal(ml):
    return 1 + ml / 100 if ml > 0 else 1 + 100 / -ml


def load_upcoming_odds(db_path=DB_PATH):
    init_db(db_path)
    today = datetime.now(ET).date().isoformat()
    con = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM latest_moneylines WHERE game_date >= ? ORDER BY game_date", con, params=(today,))
    con.close()
    return df


def compare_to_market(odds, probs, min_ev=0.03, max_edge=0.10, kelly_frac=0.25):
    d = odds.copy()
    d["gid"] = [f"sched_{r.game_date}_{r.away_team}_{r.home_team}" for r in d.itertuples()]
    d = d.merge(probs.rename(columns={"GAME_ID": "gid"}), on="gid", how="left")
    ih, ia = d.home_ml.map(american_to_prob), d.away_ml.map(american_to_prob)
    d["vig"] = ih + ia - 1
    d["mkt_home"] = ih / (ih + ia)                       # de-vigged market probability
    d["edge_home"] = d["ensemble"] - d["mkt_home"]
    dh, da = d.home_ml.map(american_to_decimal), d.away_ml.map(american_to_decimal)
    d["ev_home"] = d["ensemble"] * dh - 1
    d["ev_away"] = (1 - d["ensemble"]) * da - 1
    side_home = d.ev_home >= d.ev_away
    d["side"] = np.where(side_home, d.home_team, d.away_team)
    d["ev"] = np.where(side_home, d.ev_home, d.ev_away)
    dec = np.where(side_home, dh, da)
    d["kelly"] = (kelly_frac * d.ev / (dec - 1)).clip(lower=0)
    big = d.edge_home.abs() > max_edge
    d["flag"] = np.where(d.ensemble.isna(), "no model prob (already played?)",
                np.where(d.ev < min_ev, "", np.where(big, "CHECK (edge too big)", "BET")))
    cols = ["game_date", "away_team", "home_team", "away_ml", "home_ml", "mkt_home", "ensemble",
            "edge_home", "side", "ev", "kelly", "flag"] + [c for c in ENSEMBLE if c in d.columns]
    return d[cols]


def compare_cli(min_ev=0.03, max_edge=0.10, n_seasons=4):
    odds = load_upcoming_odds()
    if odds.empty:
        raise SystemExit("No upcoming moneylines in the DB. Run moneyline_scraper.py first.")
    games, ctx = prepare(n_seasons, refresh=True, extra_scheduled=odds)
    train = games[games.y.notna()]
    upcoming = games[games.GAME_ID.str.startswith("sched_")]
    probs = predict_probs(upcoming, ctx, train)
    res = compare_to_market(odds, probs, min_ev, max_edge)
    OUT_DIR.mkdir(exist_ok=True)
    f = OUT_DIR / f"compare_{datetime.now(ET):%Y-%m-%d}.csv"
    res.to_csv(f, index=False)
    show = res.copy()
    for c in ("mkt_home", "ensemble", "edge_home", "ev", "kelly") + tuple(ENSEMBLE):
        if c in show:
            show[c] = show[c].round(3)
    print(show.to_string(index=False))
    print("\nsaved:", f)
    return res


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("backtest")
    b.add_argument("--seasons", type=int, default=4)
    b.add_argument("--retrain-days", type=int, default=30)
    b.add_argument("--models", nargs="*", default=None, choices=list(MODELS))
    c = sub.add_parser("compare")
    c.add_argument("--min-ev", type=float, default=0.03)
    c.add_argument("--max-edge", type=float, default=0.10)
    c.add_argument("--seasons", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "backtest":
        backtest_cli(a.seasons, a.retrain_days, a.models)
    else:
        compare_cli(getattr(a, "min_ev", 0.03), getattr(a, "max_edge", 0.10), getattr(a, "seasons", 4))


if __name__ == "__main__":
    main()