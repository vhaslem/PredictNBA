"""
moneyline_scraper.py

Pulls NBA moneylines from Sports Bet Montana and upserts them into SQLite.

Schedule (cron, 11:00 daily):
    0 11 * * * cd /path/to/project && /path/to/python moneyline_scraper.py >> scrape.log 2>&1

Setup:
    pip install playwright nba_api
    playwright install chromium

First run: use `python moneyline_scraper.py --dump --headed`. That saves every JSON
response and the page text to ./debug/ so the parser can be checked against the real
site. Exit code is 1 when zero rows are parsed, so cron/monitoring can catch breakage.
"""
import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from nba_api.stats.static import teams as nba_teams

URL = "https://www.sportsbetmontana.com/en/prematch/basketball/nba"
DB_PATH = Path("nba_odds.db")
DEBUG_DIR = Path("debug")
ET = ZoneInfo("America/New_York")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

# ---------------------------------------------------------------- team names
_TEAMS = nba_teams.get_teams()
_ALIASES = {"blazers": "POR", "sixers": "PHI", "cavs": "CLE", "wolves": "MIN", "mavs": "DAL"}
_BY_ABBR = {t["abbreviation"]: t for t in _TEAMS}


def resolve_team(text):
    """Return the nba_api team dict for a string, or None if zero or multiple teams match."""
    if not isinstance(text, str) or not text or len(text) > 60:
        return None
    t = " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())
    hits = set()
    for tm in _TEAMS:
        for n in (tm["full_name"].lower(), tm["nickname"].lower()):
            if re.search(rf"\b{re.escape(n)}\b", t):
                hits.add(tm["abbreviation"])
        if t == tm["abbreviation"].lower():
            hits.add(tm["abbreviation"])
    for alias, ab in _ALIASES.items():
        if re.search(rf"\b{alias}\b", t):
            hits.add(ab)
    return _BY_ABBR[hits.pop()] if len(hits) == 1 else None


# ---------------------------------------------------------------- odds helpers
def to_american(v, key=""):
    """Coerce a price to American odds. Returns None when the format is ambiguous."""
    k = key.lower()
    try:
        if isinstance(v, dict):
            for kk in ("american", "americanOdds", "us"):
                if kk in v:
                    return to_american(v[kk], "american")
            for kk in ("decimal", "decimalOdds"):
                if kk in v:
                    return to_american(v[kk], "decimal")
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, str):
            s = v.strip().upper()
            if s in ("EVEN", "EV", "PK"):
                return 100
            if re.fullmatch(r"[+-]\d{3,4}", s):
                return int(s)
            if re.fullmatch(r"\d+\.\d+", s):
                v = float(s)
            else:
                return None
        x = float(v)
    except (ValueError, TypeError):
        return None
    if "american" in k:
        return int(x) if abs(x) >= 100 else None
    if "decimal" in k or (not x.is_integer() and 1.01 < x < 30):
        if x <= 1.0:
            return None
        return int(round((x - 1) * 100)) if x >= 2 else int(round(-100 / (x - 1)))
    return None  # bare integers are ambiguous (American vs thousandths-decimal); skip


def implied(ml):
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def valid_pair(a, h):
    """Two-way moneyline sanity check: overround must look like a real two-way market."""
    if a is None or h is None:
        return False
    if (a, h) == (-110, -110):  # almost certainly a spread/total pair, not a moneyline
        return False
    return 1.0 <= implied(a) + implied(h) <= 1.20


# ---------------------------------------------------------------- database
def init_db(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS moneylines (
        snapshot_date TEXT NOT NULL,   -- ET date of the run
        scraped_at    TEXT NOT NULL,   -- UTC timestamp of the run
        game_date     TEXT NOT NULL,   -- ET date of tip-off (ET)
        event_start   TEXT,
        away_team     TEXT NOT NULL,   -- NBA abbreviation
        home_team     TEXT NOT NULL,
        away_ml       INTEGER NOT NULL,
        home_ml       INTEGER NOT NULL,
        source        TEXT,
        PRIMARY KEY (snapshot_date, game_date, away_team, home_team)
    );
    CREATE VIEW IF NOT EXISTS latest_moneylines AS
    SELECT m.* FROM moneylines m
    JOIN (SELECT game_date, away_team, home_team, MAX(scraped_at) AS ts
          FROM moneylines GROUP BY game_date, away_team, home_team) l
      ON m.game_date = l.game_date AND m.away_team = l.away_team
     AND m.home_team = l.home_team AND m.scraped_at = l.ts;
    """)
    con.commit()
    con.close()


def save_moneylines(rows, db_path=DB_PATH):
    """Upsert one snapshot. Re-running on the same day overwrites that day's row."""
    init_db(db_path)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    snap = datetime.now(ET).date().isoformat()
    con = sqlite3.connect(db_path)
    con.executemany(
        "INSERT OR REPLACE INTO moneylines VALUES (?,?,?,?,?,?,?,?,?)",
        [(snap, now, r["game_date"], r.get("event_start"), r["away"], r["home"],
          r["away_ml"], r["home_ml"], r["source"]) for r in rows])
    con.commit()
    con.close()
    return len(rows)


# ---------------------------------------------------------------- JSON parsing
_ODDS_KEYS = ("oddsAmerican", "americanOdds", "american", "price", "odds", "decimalOdds", "decimal")
_START_KEYS = ("startTime", "start_time", "startDate", "eventStart", "start", "date", "commence_time")
_ML_LABEL = re.compile(r"money\s*-?\s*line|match\s*winner|to\s*win|\bwinner\b|2\s*way", re.I)


def _iter_dicts(o):
    if isinstance(o, dict):
        yield o
        for v in o.values():
            yield from _iter_dicts(v)
    elif isinstance(o, list):
        for v in o:
            yield from _iter_dicts(v)


def _teams_in(o, memo):
    """Ordered distinct team abbreviations found in string values under o."""
    k = id(o)
    if k in memo:
        return memo[k]
    found = []
    if isinstance(o, str):
        t = resolve_team(o)
        found = [t["abbreviation"]] if t else []
    else:
        kids = o.values() if isinstance(o, dict) else (o if isinstance(o, list) else ())
        for v in kids:
            for a in _teams_in(v, memo):
                if a not in found:
                    found.append(a)
    memo[k] = found
    return found


def _find_events(o, memo):
    """Yield the outermost sub-objects that mention exactly two NBA teams (= one event)."""
    teams = _teams_in(o, memo)
    if len(teams) < 2:
        return
    if len(teams) == 2:
        yield o
        return
    kids = list(o.values()) if isinstance(o, dict) else (list(o) if isinstance(o, list) else [])
    for c in kids:
        yield from _find_events(c, memo)


def _odds_for(o, abbr):
    for d in _iter_dicts(o):
        names = [v for v in d.values() if isinstance(v, str)]
        t = [resolve_team(s) for s in names]
        if any(x and x["abbreviation"] == abbr for x in t):
            for k in _ODDS_KEYS:
                if k in d:
                    a = to_american(d[k], k)
                    if a is not None:
                        return a
    return None


def _ml_pair(event, away, home, memo):
    cands = [d for d in _iter_dicts(event)
             if any(isinstance(v, str) and _ML_LABEL.search(v) for v in d.values())
             and {away, home} <= set(_teams_in(d, memo))]
    cands.sort(key=lambda d: sum(1 for _ in _iter_dicts(d)))
    for m in cands:
        a, h = _odds_for(m, away), _odds_for(m, home)
        if valid_pair(a, h):
            return a, h
    return None


def _start_et(event):
    """First recognisable start time anywhere in the event subtree, converted to ET."""
    for d in _iter_dicts(event):
        for k in _START_KEYS:
            v = d.get(k)
            try:
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 1e9:
                    ts = datetime.fromtimestamp(v / 1000 if v > 1e12 else v, tz=timezone.utc)
                elif isinstance(v, str) and re.match(r"\d{4}-\d{2}-\d{2}", v):
                    ts = datetime.fromisoformat(v.replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                else:
                    continue
                return ts.astimezone(ET)
            except (ValueError, OSError):
                continue
    return None


def _home_away(event, order):
    """Explicit home/away keys if present, else assume US convention: first listed = away."""
    if isinstance(event, dict):
        h = next((resolve_team(event[k]) for k in ("homeTeam", "home", "homeName") if isinstance(event.get(k), str)), None)
        a = next((resolve_team(event[k]) for k in ("awayTeam", "away", "visitor", "awayName") if isinstance(event.get(k), str)), None)
        if h and a:
            return a["abbreviation"], h["abbreviation"]
    return order[0], order[1]


def parse_json_payloads(payloads):
    rows, seen = [], set()
    today = datetime.now(ET).date().isoformat()
    for p in payloads:
        memo = {}
        for ev in _find_events(p["data"], memo):
            away, home = _home_away(ev, _teams_in(ev, memo))
            pair = _ml_pair(ev, away, home, memo)
            if not pair:
                continue
            start = _start_et(ev)
            gd = start.date().isoformat() if start else today
            if (gd, away, home) in seen:
                continue
            seen.add((gd, away, home))
            rows.append({"game_date": gd, "event_start": start.isoformat() if start else None,
                         "away": away, "home": home, "away_ml": pair[0], "home_ml": pair[1],
                         "source": "json"})
    return rows


# ---------------------------------------------------------------- page-text parser (primary)
# Layout observed on the live page (one game):
#     <away team>
#     <home team>
#     Nov 19 6:00 PM              <- site local time (Mountain)
#     +220                        <- "more markets" count, NOT a price
#     -120   +102                 <- moneyline: away, home
#     -1.5 -109 +1.5 -109         <- spread: handicap + price, per team
SITE_TZ = ZoneInfo("America/Denver")
_DATE_LINE = re.compile(r"^([A-Z][a-z]{2})\s+(\d{1,2})\s+(\d{1,2}:\d{2}\s*[AP]M)$")
_INT_ODDS = re.compile(r"^[+-]\d{1,4}$")  # 1-2 digit values are the markets count
_SPREAD = re.compile(r"^[+-]\d+\.\d$")


def _tipoff_et(mon, day, clock):
    now = datetime.now(ET)
    dt = datetime.strptime(f"{mon} {day} {now.year} {clock.upper().replace(' ', '')}", "%b %d %Y %I:%M%p")
    dt = dt.replace(tzinfo=SITE_TZ).astimezone(ET)
    if (now - dt).days > 180:  # Dec page scraped in Jan, etc.
        dt = dt.replace(year=dt.year + 1)
    return dt


def parse_page_text(text):
    """Parse the visible page text. Only matches 'team, team, date' triples, so the futures list is ignored."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    rows = []
    for i in range(len(lines) - 2):
        away = resolve_team(lines[i]) if len(lines[i]) < 40 else None
        home = resolve_team(lines[i + 1]) if len(lines[i + 1]) < 40 else None
        m = _DATE_LINE.match(lines[i + 2])
        if not (away and home and m):
            continue
        ints = []
        for tok in lines[i + 3:i + 14]:
            if _SPREAD.match(tok):
                break  # moneyline columns end where the spread handicap starts
            if _INT_ODDS.match(tok):
                ints.append(int(tok))
            else:
                break
        if len(ints) < 2 or min(abs(ints[-2]), abs(ints[-1])) < 100 or not valid_pair(ints[-2], ints[-1]):
            continue  # ints[0] is the markets count when present; ML is always the last two
        t = _tipoff_et(*m.groups())
        rows.append({"game_date": t.date().isoformat(), "event_start": t.isoformat(),
                     "away": away["abbreviation"], "home": home["abbreviation"],
                     "away_ml": ints[-2], "home_ml": ints[-1], "source": "text"})
    return rows


# ---------------------------------------------------------------- browser
def fetch_page(headed=False, wait_s=8):
    from playwright.sync_api import sync_playwright
    payloads = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed)
        ctx = browser.new_context(user_agent=UA, locale="en-US", timezone_id="America/Denver",
                                  viewport={"width": 1440, "height": 2200})
        page = ctx.new_page()

        def on_response(r):
            try:
                if "json" in (r.headers.get("content-type") or ""):
                    payloads.append({"url": r.url, "data": r.json()})
            except Exception:
                pass

        page.on("response", on_response)
        page.goto(URL, wait_until="networkidle", timeout=90_000)
        page.wait_for_timeout(wait_s * 1000)
        for _ in range(8):  # trigger lazy-loaded events
            page.mouse.wheel(0, 2500)
            page.wait_for_timeout(600)
        text = page.inner_text("body")
        browser.close()
    return payloads, text


def scrape_moneylines(db_path=DB_PATH, dump=False, headed=False):
    """Scrape, validate and store today's moneylines. Returns number of games saved."""
    payloads, text = fetch_page(headed=headed)
    if dump:
        DEBUG_DIR.mkdir(exist_ok=True)
        (DEBUG_DIR / "page_text.txt").write_text(text)
        index = []
        for i, p in enumerate(payloads):
            (DEBUG_DIR / f"payload_{i:03d}.json").write_text(json.dumps(p["data"]))
            index.append({"i": i, "url": p["url"]})
        (DEBUG_DIR / "index.json").write_text(json.dumps(index, indent=1))
    rows = parse_page_text(text) or parse_json_payloads(payloads)
    n = save_moneylines(rows, db_path) if rows else 0
    print(f"{datetime.now(ET):%Y-%m-%d %H:%M} ET  payloads={len(payloads)}  games_saved={n}")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--dump", action="store_true", help="save raw JSON + page text to ./debug")
    ap.add_argument("--headed", action="store_true", help="show the browser (helps with bot detection)")
    a = ap.parse_args()
    sys.exit(0 if scrape_moneylines(Path(a.db), a.dump, a.headed) > 0 else 1)