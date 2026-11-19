Exploratory analysis of methods to predict the outcomes of NBA games and their effects. I do not condone or support gambling in any fashion.

# functions
## TESTING: compare_kalshi.py & kalshi_fetch.py
Pulls odds from kalshi to account for public sentiment, compares model edge.
*python kalshi_fetch.py* fetches Kalshi data in similar fashion
*python compare_kalshi.py compare* evaluates v2 model against kalshi book to assess for gaps

Trading volume seems suspicious, highly skeptical of the entire Kalshi platform in general. As Kalshi bans continue and it becomes less representative of sentiment, I find this less compelling as an area to research.
## moneyline_scraper.py
Pulls daily odds from Montana SportsBet and places in SQL table. 

*python moneyline_scraper.py --dump --headed* must be run before *compare* is run.

## nba_model_v2.py
New model, accounts for injuries. Kalman drift influenced team ratings, adjusts for opponent strength, and uses ridge regression to estimate player value and build lineup strength.

*python nba_model_v2.py backtest* evaluates against past nba data to calculate model performance metrics
*python nba_model.py compare [--out "Player Name"]* pulls moneyline and evaluates models, outputs estimated probabilities. Optional flag allows for injury reports, specify dropped players to correctly update lineup strength.

v2 backtested against v1 has about a 1% log-loss edge on original model.

## nba_model.py
Original model for ensemble metric evaluation.

*python nba_model.py backtest* evaluates against past nba data to calculate model performance metrics
*python nba_model.py compare* pulls moneyline and evaluates models, outputs estimated probabilities, edges, EV, kelly. CHECK implies >10 pt edge so mark these as suspicious, model could be failing to pick up on key game factors that modify odds.
