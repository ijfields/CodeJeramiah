Brief status: I’ll give you a single “start-here” document you can drop into the new repo root as README.md (or START_HERE.md). It reflects subagents (Option A), Strategy Spec v1, Tradier + TradeLocker/Trend, Polygon flat files, logs/KPIs, and scales to Option C later.

### Suggested file: README.md

```markdown
## Algo Platform — Start Here

A spec-first, multi-agent platform that designs, backtests, and deploys trading strategies across options, futures (micros), and forex using Lumibot. Subagents coordinate research → spec → code → backtest → evaluate → fix → deploy. Built to scale from simple file orchestration to a full DAG/evaluation harness.

- Agent role pattern inspiration: contains-studio/agents (subagent definitions) [link](https://github.com/contains-studio/agents)
- OpenAI Agents SDK for Python [link](https://openai.github.io/openai-agents-python/agents/)
- Claude Code subagents (YAML markdown agents) [link](https://docs.anthropic.com/en/docs/claude-code/sub-agents)
- Engineering role example: AI Engineer [link](https://github.com/contains-studio/agents/blob/main/engineering/ai-engineer.md)
- Tradier futures margins reference (feature-gated at start) [link](https://tradier.com/individuals/pricing/futures-margins)

### 1) Architecture (Option A, scalable to Option C)
- Subagents: Researcher → Strategy Architect → Lumibot Coder → Backtest Runner → Evaluator → Fixer → Deployer
- Artifacts per run: strategy_spec.json, lumibot_strategy.py, runner config, results (settings.json, tearsheet.csv/html, trades.csv/html, stats.csv)
- Scale path: keep same artifacts and add a DAG that runs grids, regression tests, and promotion gates

### 1a) RBI lifecycle and subagents (explicit)
- RBI stands for Research → Backtest → Implement.
- Lifecycle mapping:
  - Research (R): `agents/researcher.py` gathers ideas/data and writes research notes
  - Spec & Code: `agents/architect.py` creates `strategy_spec.json`; `agents/coder_lumibot.py` emits `lumibot_strategy.py`
  - Backtest (B): `agents/runner.py` executes backtests, writes artifacts to `strategies/{slug}/results/{run_id}`
  - Evaluate & Fix: `agents/evaluator.py` scores KPIs; `agents/fixer.py` proposes parameter/code changes
  - Implement (I): `agents/deployer.py` promotes to demo/live when KPIs sustain and approvals are met
  - Observability: dashboard indexes runs and shows KPIs/promotion status

### 1b) Agent setup checklist (project-level)
- Create Claude subagents: `.claude/agents/architect.md`, `.claude/agents/coder-reviewer.md`, `.claude/agents/debugger-fixer.md`, `.claude/agents/test-runner.md`.
- Add Python execution agents (OpenAI Agents SDK shims): `agents/coordinator.py`; ensure `agents/runner.py`, `agents/evaluator.py`, `agents/fixer.py`, `agents/deployer.py` are present.
- Initialize tasks store: `ops/tasks.json` with `{ "tasks": [] }`.
- Validate `strategy_spec.json` files against the v1 schema before runs.
- Route models per role to control cost (cheaper for Researcher/Evaluator; strongest for Coder/Fixer).

### 2) Strategy Spec v1 (JSON, single source of truth)
- Core keys
  - name, slug, universe, timeframe
  - entry: signal_logic, timing, triggers
  - legs: option/future/forex legs; right/side/delta/dte/width/qty
  - risk_exits: stop/tp/trailing/time-exit/roll
  - sizing: per-trade risk or debit/contract caps
  - data: source=polygon_flat; paths for equities_min, equities_5min, options_min; tz
  - broker: tradier | tradelocker (TrendMarkets); account_type (paper|live)
  - backtest: start, end, cash, fees, benchmark, outputs
- Minimal skeleton:
```json
{
  "version": "1.0",
  "name": "Strategy Name",
  "slug": "strategy_slug",
  "universe": ["SPY"],
  "timeframe": "1min",
  "entry": { "signal_logic": [], "enter_time": "09:45:00" },
  "legs": [],
  "risk_exits": {},
  "sizing": {},
  "data": {
    "source": "polygon_flat",
    "paths": { "equities_min": "data/processed/equities_min/{YYYY}/{MM}/{YYYY}-{MM}-{DD}.parquet",
               "equities_5min": "data/processed/equities_5min/{YYYY}/{MM}/{YYYY}-{MM}-{DD}.parquet",
               "options_min": "data/processed/options_min/{YYYY}/{MM}/{YYYY}-{MM}-{DD}.parquet" },
    "tz": "America/New_York"
  },
  "broker": { "name": "tradier", "account_type": "paper" },
  "backtest": { "start": "2023-01-01", "end": "2024-01-01", "cash": 100000, "fees": { "percent_fee": 0.001 } },
  "outputs": ["settings.json","tearsheet.csv","tearsheet.html","trades.csv","trades.html","stats.csv"]
}
```

### 3) Repo layout
- agents/ researcher.py, architect.py, coder_lumibot.py, runner.py, evaluator.py, fixer.py, deployer.py
- strategies/{slug}/ strategy_spec.json, lumibot_strategy.py, runner.py, results/, logs/
- libs/
  - brokers/ tradier_adapter.py, tradelocker_adapter.py
  - lumibot_helpers/ leg_builder.py, order_factories.py, polygon_io.py, metrics.py, io.py
- data/ processed/ equities_min|equities_5min|options_min/…
- logs/ aggregated logs and tearsheets (ingested to runs table)
- credentials.py, .env, requirements.txt

### 4) Setup
- Python 3.11+ recommended
- Install
```bash
pip install -r requirements.txt
```
- .env (example)
```bash
OPENAI_API_KEY=...
GOOGLE_API_KEY=...
TRADIER_API_KEY=...
TRADIER_ACCOUNT_ID=...
TRADELOCKER_ENV=https://demo.tradelocker.com
TRADELOCKER_USERNAME=...
TRADELOCKER_PASSWORD=...
TRADELOCKER_SERVER=...
IS_BACKTESTING=true
DATA_ROOT=./data/processed
DISCORD_WEBHOOK_URL=
ACCOUNT_HISTORY_DB_CONNECTION_STR=
```
- LLM models: To use Gemini, set GOOGLE_API_KEY and configure models (main=gemini-1.5-pro-latest, research=gemini-1.5-flash-latest).
- credentials.py (stub)
```python
import os
IS_BACKTESTING = os.getenv("IS_BACKTESTING", "true").lower() == "true"

TRADIER_CONFIG = {
  "API_KEY": os.getenv("TRADIER_API_KEY"),
  "ACCOUNT_ID": os.getenv("TRADIER_ACCOUNT_ID"),
}

TRADELOCKER_CONFIG = {
  "ENVIRONMENT": os.getenv("TRADELOCKER_ENV", "https://demo.tradelocker.com"),
  "USERNAME": os.getenv("TRADELOCKER_USERNAME"),
  "PASSWORD": os.getenv("TRADELOCKER_PASSWORD"),
  "SERVER": os.getenv("TRADELOCKER_SERVER"),
}
```

### 5) Broker adapters
- Tradier adapter
  - Options now; Futures behind feature flag until enabled per your account (micros: MES, MNQ, M2K, MGC, MYM)
  - Backlog: IBKR adapter for futures/options
- TradeLocker adapter (single adapter; TrendMarkets as broker)
  - Parametrize env/server; supports forex symbols (e.g., XAUUSD, EURUSD, NAS100, US30)

### 6) Data (Polygon flat files)
- Backtests read minute/5-minute parquet via libs/lumibot_helpers/polygon_io.py
- Ensure columns: open, high, low, close, volume, timestamp (tz-aware NY time preferred)
- Options parquet: must include ticker, underlying, expiration, close (minute)

### 7) Seed strategies
- ZRCE Second Chance (SPY options): zone-based entry, second-chance confirmation, TP/SL/trailing/time-exit
- Ghetto Spread (AMD calls): staged debit spread (long call → convert by selling OTM call on trigger)
- Later: “Signals” family from kirkland_signals.csv (forex/futures indices via TradeLocker/Trend)

### 8) Run flows
- Generate/curate spec → agents/runner.py executes backtest
```bash
python agents/runner.py --spec strategies/zrce_second_chance/strategy_spec.json
```
- One-command RBI workflow (research→spec→code→backtest→evaluate→fix/deploy):
```bash
python tools/rbi.py --slug sma_demo --demo --symbol SPY
```
- Outputs saved in strategies/{slug}/results/{run_id}/ with:
  - settings.json, tearsheet.csv/html, trades.csv/html, stats.csv
- Evaluator checks KPIs:
  - max_drawdown <= 20%, sharpe >= 0.8, win_rate >= 55%, trades >= 100
- Promotion:
  - If KPIs sustained over a rolling window (e.g., 20 trading days), mark for demo
  - Live requires manual approval

### 9) Logs and observability
- Ingest lumibot outputs including trades.html
- Normalize filenames to a runs registry (CSV/SQLite/Parquet)
- Minimal dashboard (Streamlit or FastAPI+simple UI) to list runs, KPIs, drill to trades/tearsheet

### 10) Roadmap (extracts)
- Short term
  - Finalize Strategy Spec v1 + subagent stubs (OpenAI Agent SDK)
  - Polygon parquet loaders; ZRCE + Ghetto Spread live in repo
  - Tradier options, TradeLocker/Trend forex adapter
- Mid term
  - Tradier futures feature flag → enable when account ready
  - Signals ingestion from kirkland_signals.csv (forex/futures micros)
  - Promotion gates and basic scheduling for demo
- Long term (Option C)
  - DAG + grid backtests + regression suite
  - IBKR adapter
  - Advanced UI and multi-strategy portfolio evaluation

### 11) Contribution and governance
- Specs are canonical; code must be generated/edited to conform
- Every run persists artifacts; reproducibility required before promotion

### 12) Disclaimers
- Trading involves substantial risk. Use paper/demo until strategies meet KPIs and pass review.
```

If you prefer, save as START_HERE.md and we’ll keep README.md concise with a link.

- Document includes subagent flow, Strategy Spec v1 skeleton, repo layout, setup, adapters (Tradier, TradeLocker/Trend), Polygon flat files, logs/KPIs, seed strategies, roadmap, and links to contains-studio/agents and Tradier futures margins.
- Ready to scaffold the repo using this as the root README.