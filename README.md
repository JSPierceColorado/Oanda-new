# FX Strategy Generator

This service evolves rule-based FX strategies from the `history` tab in your Google Sheet and writes live entry signals into a `Signals` tab.

## What it does

- Reads `Screener` for the live feature set.
- Reads `history` for historical snapshots.
- Evolves weighted long/short strategies using win rate + net profitability fitness.
- Saves the current best strategy locally and into `BestStrategy` / `StrategyLeaderboard`.
- Scores every live FX pair from `-100` (strong short) to `100` (strong long).
- Writes current signals into the `Signals` sheet tab.

## Required environment variables

- `SHEET_ID`
- `GOOGLE_SERVICE_ACCOUNT_JSON_B64` or `GOOGLE_SERVICE_ACCOUNT_JSON` or `GOOGLE_APPLICATION_CREDENTIALS`

## Optional environment variables

- `SCREENER_TAB` default `Screener`
- `HISTORY_TAB` default `history`
- `SIGNALS_TAB` default `Signals`
- `LEADERBOARD_TAB` default `StrategyLeaderboard`
- `BEST_STRATEGY_TAB` default `BestStrategy`
- `SIGNAL_POLL_SECONDS` default `60`
- `RETRAIN_MINUTES` default `60`
- `POPULATION_SIZE` default `80`
- `GENERATIONS` default `30`
- `ELITE_COUNT` default `10`
- `MUTATION_RATE` default `0.18`
- `MUTATION_STRENGTH` default `0.45`
- `HOLD_MIN_STEPS` default `1`
- `HOLD_MAX_STEPS` default `8`
- `THRESHOLD_MIN` default `20`
- `THRESHOLD_MAX` default `75`
- `MIN_TRADES` default `15`
- `VALIDATION_SPLIT` default `0.25`
- `STATE_FILE` default `/tmp/best_strategy.json`
- `LOG_LEVEL` default `INFO`

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8080
```

## Railway deploy

Push this repo to GitHub, create a Railway project, deploy from the repo, and set the environment variables. Railway will use the root `Dockerfile`.

## Endpoints

- `GET /health`
- `GET /status`
- `POST /train`
- `POST /score`
- `POST /cycle`
