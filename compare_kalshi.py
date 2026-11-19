"""
compare_kalshi.py

Compares the v2 model against Kalshi's price (the "crowd" probability) and, when available, the
de-vigged sportsbook line, for the same games at the same moment.

    python compare_kalshi.py compare [--out "Player Name" ...] [--as-of 2026-11-19T15:00:00+00:00]
    python compare_kalshi.py score                 # after games finish: who was closer to reality?

Reading the output:
  * Kalshi is not a poll. It is a money-weighted order book, so it already contains injury news,
    sharp money and the exchange's own spread. A gap to the model is a hypothesis to test,
    not an opportunity.
  * Executable price matters. You buy at the ask and pay a fee, so edge is measured against
    ask + fee, not against the mid.
  * `compare` appends each prediction to a `model_preds` table. `score` joins those to results so
    the question "is the model ever closer than Kalshi?" gets answered with data.
"""
import argparse
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

import nba_model as v1
import nba_model_v2 as v2
from kalshi_fetch import kalshi_game_probs
from moneyline_scraper import DB_PATH, init_db

FEE_RATE = 0.07          # taker fee ~ FEE_RATE * p * (1-p) per $1 contract. CHECK Kalshi's current fee schedule.
KEY = ["game_date", "away_team", "home_team"]


def kalshi_fee(c):
    return FEE_RATE * c * (1 - c)


def _book_probs(db_path):
    """De-vigged sportsbook home probability from the scraper's latest_moneylines view."""
    init_db(db_path)
    con = sqlite3.connect(db_path)
    o = pd.read_sql_query("SELECT game_date, away_team, home_team, away_ml, home_ml FROM latest_moneylines", con)
    con.close()
    if o.empty:
        return pd.DataFrame(columns=KEY + ["p_book"])
    imp = lambda ml: np.where(ml > 0, 100 / (ml + 100), -ml / (-ml + 100))
    ih, ia = imp(o.home_ml), imp(o.away_ml)
    o["p_book"] = ih / (ih + ia)
    return o[KEY + ["p_book"]]


def _save_preds(res, db_path):
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE IF NOT EXISTS model_preds (
        snapshot_ts TEXT NOT NULL, game_date TEXT, away_team TEXT, home_team TEXT,
        p_v2 REAL, k_home REAL, p_book REAL,
        PRIMARY KEY (snapshot_ts, game_date, away_team, home_team))""")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con.executemany("INSERT OR REPLACE INTO model_preds VALUES (?,?,?,?,?,?,?)",
                    [(ts, r.game_date, r.away_team, r.home_team, r.p_v2,
                      None if pd.isna(r.k_home) else r.k_home,
                      None if pd.isna(r.p_book) else r.p_book) for r in res.itertuples()])
    con.commit()
    con.close()


def compare(db_path=DB_PATH, as_of=None, out_names=(), max_spread=6.0, min_ev=0.03, max_gap=0.10, n_seasons=8):
    k = kalshi_game_probs(db_path, as_of=as_of, max_spread=max_spread)
    if k.empty:
        raise SystemExit("No Kalshi snapshots found. Run kalshi_fetch.py first.")
    if not out_names:
        print("NOTE: no --out players given; the model assumes everyone is healthy, Kalshi does not.\n")

    games, ctx = v2.prepare_v2(n_seasons, True, k[KEY].drop_duplicates(), out_names)
    train = games[games.y.notna()]
    upcoming = games[games.GAME_ID.str.startswith("sched_")]
    p = v1.predict_probs(upcoming, ctx, train, model_names=v2.V2_ENS)[["GAME_ID", "ensemble"]]
    p = p.rename(columns={"ensemble": "p_v2"})
    parts = p.GAME_ID.str.split("_", expand=True)
    p["game_date"], p["away_team"], p["home_team"] = parts[1], parts[2], parts[3]

    res = k.merge(p.drop(columns="GAME_ID"), on=KEY, how="inner").merge(_book_probs(db_path), on=KEY, how="left")
    res["gap_v2_kalshi"] = res.p_v2 - res.k_home           # + : model likes the home team more than Kalshi
    res["gap_kalshi_book"] = res.k_home - res.p_book       # + : Kalshi likes the home team more than the books

    # executable edge on Kalshi: buy that team's YES at the ask, pay the fee
    def ev(prob, ask_cents):
        c = ask_cents / 100
        cost = c + kalshi_fee(c)
        return np.where((ask_cents > 0) & (ask_cents < 100), prob / cost - 1, np.nan)

    res["ev_home"] = ev(res.p_v2, res.yes_ask_h.astype(float))
    res["ev_away"] = ev(1 - res.p_v2, res.yes_ask_a.astype(float))
    best_home = np.nan_to_num(res.ev_home, nan=-9) >= np.nan_to_num(res.ev_away, nan=-9)
    res["side"] = np.where(best_home, res.home_team, res.away_team)
    res["ev"] = np.where(best_home, res.ev_home, res.ev_away)
    res["flag"] = np.where(res.k_home.isna(), "no usable Kalshi price (wide/empty book)",
                  np.where(res.ev.isna() | (res.ev < min_ev), "",
                  np.where(res.gap_v2_kalshi.abs() > max_gap, "CHECK (gap too big)", "BET")))
    _save_preds(res, db_path)

    show = res[KEY + ["p_v2", "k_home", "p_book", "gap_v2_kalshi", "gap_kalshi_book", "side", "ev", "k_overround",
                      "volume_h", "flag"]].round(3)
    v1.OUT_DIR.mkdir(exist_ok=True)
    f = v1.OUT_DIR / f"kalshi_compare_{datetime.now(v1.ET):%Y-%m-%d}.csv"
    show.to_csv(f, index=False)
    print(show.to_string(index=False))
    print(f"\nmean |v2 - Kalshi| = {res.gap_v2_kalshi.abs().mean():.3f}   saved: {f}")
    return res


def score(db_path=DB_PATH):
    """For finished games: log loss of v2, Kalshi and the sportsbook on identical games."""
    con = sqlite3.connect(db_path)
    try:
        mp = pd.read_sql_query("SELECT * FROM model_preds", con)
    except Exception:
        raise SystemExit("No predictions logged yet. Run `compare` daily first.")
    finally:
        con.close()
    mp = mp.sort_values("snapshot_ts").groupby(KEY).tail(1)           # last prediction before the game
    tl = v1.build_team_log(v1.load_game_logs(2))
    res = tl[tl.is_home == 1][["GAME_DATE", "TEAM_ABBREVIATION", "won"]].copy()
    res["game_date"] = res.GAME_DATE.dt.strftime("%Y-%m-%d")
    res = res.rename(columns={"TEAM_ABBREVIATION": "home_team", "won": "y"})
    d = mp.merge(res[["game_date", "home_team", "y"]], on=["game_date", "home_team"]).dropna(subset=["p_v2", "k_home"])
    if len(d) < 30:
        raise SystemExit(f"Only {len(d)} finished games with both predictions. Too few to say anything.")

    ll = lambda y, p: -(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(p, 1e-4, 1 - 1e-4)))
    y = d.y.to_numpy()
    out = {"v2": ll(y, d.p_v2.to_numpy()), "kalshi": ll(y, d.k_home.to_numpy())}
    if d.p_book.notna().all():
        out["sportsbook"] = ll(y, d.p_book.to_numpy())
    print(f"{len(d)} finished games")
    for name, v in out.items():
        print(f"  {name:<10} log loss {v.mean():.4f}")
    gain = out["kalshi"] - out["v2"]                                    # + : v2 closer to reality
    rng = np.random.default_rng(0)
    boots = np.array([gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(3000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"v2 vs Kalshi gain {gain.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  ->",
          "v2 better" if lo > 0 else "Kalshi better" if hi < 0 else "no demonstrated difference")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--out", nargs="*", default=[])
    c.add_argument("--as-of", default=None, help="UTC ISO time; uses the latest Kalshi snapshot at or before it")
    c.add_argument("--max-spread", type=float, default=6.0, help="widest bid-ask (cents) still trusted as a price")
    c.add_argument("--min-ev", type=float, default=0.03)
    c.add_argument("--max-gap", type=float, default=0.10)
    c.add_argument("--seasons", type=int, default=8)
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "compare":
        compare(a.db, a.as_of, a.out, a.max_spread, a.min_ev, a.max_gap, a.seasons)
    else:
        score(a.db)


if __name__ == "__main__":
    main()