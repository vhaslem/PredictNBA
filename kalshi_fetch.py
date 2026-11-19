"""
kalshi_fetch.py

Snapshots Kalshi's NBA game-winner markets (series KXNBAGAME) into the same SQLite DB as the
sportsbook odds, and turns them into one win probability per game.

Why snapshots, not one daily pull: a Kalshi price is a live order book, not a posted line. It
drifts all day (injury news, money), so we store every pull and later pick the snapshot that
matches the moment we compare against.

Schedule (ET machine clock, every 30 min from 10:00 to 23:30):
    */30 10-23 * * * cd /path/to/project && python kalshi_fetch.py >> kalshi.log 2>&1

Public market data needs no API key. Event tickers look like KXNBAGAME-26APR25NYKATL:
date (YYMonDD) + AWAY + HOME. Each event has two markets, one per team ("will <team> win?").
"""
import argparse
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from nba_api.stats.static import teams as nba_teams

from moneyline_scraper import DB_PATH

BASES = ["https://api.elections.kalshi.com/trade-api/v2",   # tried in order; docs list both hosts
         "https://external-api.kalshi.com/trade-api/v2"]
SERIES = "KXNBAGAME"
ET = ZoneInfo("America/New_York")
_EVENT = re.compile(rf"^{SERIES}-(\d{{2}})([A-Z]{{3}})(\d{{2}})([A-Z]{{3}})([A-Z]{{3}})$")
_MON = {m: i for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
_CODE_FIX = {"BRK": "BKN", "PHO": "PHX", "NOR": "NOP"}   # Kalshi code -> NBA code, if they ever differ
_NBA = {t["abbreviation"] for t in nba_teams.get_teams()}


# ---------------------------------------------------------------- HTTP
def _get(path, params):
    last = None
    for base in BASES:
        for attempt in range(4):
            try:
                r = requests.get(f"{base}{path}", params=params, timeout=20, headers={"Accept": "application/json"})
                if r.status_code == 429 or r.status_code >= 500:   # rate limit / server hiccup: back off
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                last = e
                time.sleep(1 + attempt)
    raise RuntimeError(f"Kalshi request failed on all hosts: {last}")


def fetch_nba_markets(status="open"):
    """All open game-winner markets, following the pagination cursor."""
    out, cursor = [], None
    while True:
        params = {"series_ticker": SERIES, "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        d = _get("/markets", params)
        out += d.get("markets", [])
        cursor = d.get("cursor")
        if not cursor:
            return out
        time.sleep(0.3)


# ---------------------------------------------------------------- parsing
def _cents(m, key):
    """Price in cents. Handles both integer-cent and '*_dollars' string fields."""
    d = m.get(f"{key}_dollars")
    if d not in (None, ""):
        try:
            return round(float(d) * 100, 2)
        except ValueError:
            pass
    v = m.get(key)
    return float(v) if v not in (None, "") else None


def parse_markets(markets):
    rows, bad = [], set()
    for m in markets:
        mt = _EVENT.match(m.get("event_ticker", ""))
        if not mt:
            continue
        yy, mon, dd, a, h = mt.groups()
        away, home = _CODE_FIX.get(a, a), _CODE_FIX.get(h, h)
        team = _CODE_FIX.get(m["ticker"].rsplit("-", 1)[-1], m["ticker"].rsplit("-", 1)[-1])
        if {away, home} - _NBA:
            bad |= {away, home} - _NBA
            continue
        if team not in (away, home):
            continue
        rows.append({"market_ticker": m["ticker"], "game_date": f"20{yy}-{_MON[mon]:02d}-{dd}",
                     "away_team": away, "home_team": home, "team": team,
                     "yes_bid": _cents(m, "yes_bid"), "yes_ask": _cents(m, "yes_ask"),
                     "last_price": _cents(m, "last_price"), "volume": m.get("volume"),
                     "open_interest": m.get("open_interest"), "status": m.get("status"),
                     "close_time": m.get("close_time")})
    if bad:
        print("WARNING: unrecognised team codes (add to _CODE_FIX):", sorted(bad))
    return rows


# ---------------------------------------------------------------- storage
def init_kalshi_db(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS kalshi_prices (
        snapshot_ts   TEXT NOT NULL,   -- UTC ISO
        game_date     TEXT NOT NULL,   -- from the event ticker
        away_team     TEXT NOT NULL,
        home_team     TEXT NOT NULL,
        team          TEXT NOT NULL,   -- the team this market pays out on
        market_ticker TEXT NOT NULL,
        yes_bid REAL, yes_ask REAL, last_price REAL,   -- cents
        volume INTEGER, open_interest INTEGER, status TEXT, close_time TEXT,
        PRIMARY KEY (snapshot_ts, market_ticker)
    );
    """)
    con.commit()
    con.close()


def snapshot(db_path=DB_PATH):
    """One pull -> one batch of rows. Returns number of games captured."""
    init_kalshi_db(db_path)
    rows = parse_markets(fetch_nba_markets())
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = sqlite3.connect(db_path)
    con.executemany(
        "INSERT OR REPLACE INTO kalshi_prices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(ts, r["game_date"], r["away_team"], r["home_team"], r["team"], r["market_ticker"],
          r["yes_bid"], r["yes_ask"], r["last_price"], r["volume"], r["open_interest"],
          r["status"], r["close_time"]) for r in rows])
    con.commit()
    con.close()
    n = len({(r["game_date"], r["away_team"], r["home_team"]) for r in rows})
    print(f"{datetime.now(ET):%Y-%m-%d %H:%M} ET  markets={len(rows)}  games={n}")
    return n


# ---------------------------------------------------------------- one probability per game
def kalshi_game_probs(db_path=DB_PATH, from_date=None, as_of=None, max_spread=6.0):
    """
    Latest snapshot at or before `as_of` (UTC ISO; default now) for each market, collapsed to a game.

    k_home = average of two estimates of P(home wins):
        mid of the home market, and 1 - mid of the away market.
    A mid is only trusted when bid and ask both exist and the spread is <= max_spread cents.
    Price is NOT averaged with last_price: a stale last trade is worse than no price.
    """
    as_of = as_of or datetime.now(timezone.utc).isoformat(timespec="seconds")
    from_date = from_date or datetime.now(ET).date().isoformat()
    init_kalshi_db(db_path)
    con = sqlite3.connect(db_path)
    d = pd.read_sql_query("SELECT * FROM kalshi_prices WHERE snapshot_ts <= ? AND game_date >= ?",
                          con, params=(as_of, from_date))
    con.close()
    if d.empty:
        return d
    d = d.sort_values("snapshot_ts").groupby("market_ticker").tail(1)
    key = ["game_date", "away_team", "home_team"]
    cols = ["yes_bid", "yes_ask", "last_price", "volume", "open_interest", "snapshot_ts"]
    h = d[d.team == d.home_team].set_index(key)[cols].add_suffix("_h")
    a = d[d.team == d.away_team].set_index(key)[cols].add_suffix("_a")
    g = h.join(a, how="outer").reset_index()

    def mid(bid, ask):
        ok = (bid > 0) & (ask < 100) & (ask - bid <= max_spread)
        return np.where(ok, (bid + ask) / 2, np.nan)

    g["mid_h"] = mid(g.yes_bid_h.astype(float), g.yes_ask_h.astype(float))
    g["mid_a"] = mid(g.yes_bid_a.astype(float), g.yes_ask_a.astype(float))
    est = np.column_stack([g.mid_h / 100, 1 - g.mid_a / 100])
    g["k_home"] = np.nanmean(est, axis=1) if len(g) else np.nan
    g["k_overround"] = g.mid_h + g.mid_a - 100          # should be near 0; large = stale/illiquid
    g["snapshot_ts"] = g[["snapshot_ts_h", "snapshot_ts_a"]].max(axis=1)
    return g


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean of all-NaN rows is expected
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--show", action="store_true", help="print per-game probabilities after the snapshot")
    a = ap.parse_args()
    n = snapshot(a.db)
    if a.show:
        print(kalshi_game_probs(a.db)[["game_date", "away_team", "home_team", "mid_h", "mid_a", "k_home",
                                       "k_overround", "volume_h", "volume_a"]].round(3).to_string(index=False))
    sys.exit(0 if n > 0 else 1)