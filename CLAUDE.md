# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A multi-agent stock trading system that uses GPT-4o-mini via the OpenAI API, with Alpaca paper trading for execution, Finnhub for news, yfinance for market data, and Opik for observability. It has two frontends: a terminal REPL (`main.py`) and a Chainlit web UI (`app.py`).

## Commands

```bash
# Web UI (primary)
chainlit run app.py

# Terminal REPL (simpler, no UI)
python main.py

# Install dependencies
pip install -r requirements.txt
```

There are no tests, linter, or build steps configured.

## Required Environment Variables (.env)

`OPENAI_API_KEY`, `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `FINNHUB_API_KEY`, `OPIK_API_KEY`

## Architecture

The system uses a two-level agent hierarchy. An **orchestrator** (defined identically in both `main.py` and `app.py`) coordinates four **specialist agents** via OpenAI tool-calling. Each specialist is exposed to the orchestrator as a callable tool.

```
User → Orchestrator → call_market_analyst(query)  → Market Analyst agent loop
                     → call_news_analyst(query)    → News Analyst agent loop
                     → call_risk_manager(query)    → Risk Manager agent loop
                     → call_trader(query)           → Trader agent loop
```

### Key flow

1. **`agent_loop.py:run_agent`** — the generic ReAct loop reused by all specialist agents. Takes a system prompt, user message, tool schemas, and tool function map. Loops up to `MAX_TURNS` calling tools until the LLM produces a text response.
2. **`agents.py`** — defines the four specialists. Each is a system prompt + a set of tools from `tools/`. The `AGENT_SCHEMAS` / `AGENT_FUNCTIONS` dicts register them as the orchestrator's "tools."
3. **`app.py`** — Chainlit frontend. Runs its own orchestrator loop (async) with a sidebar showing the live Alpaca portfolio. Specialist calls appear as expandable steps in the Chainlit UI.
4. **`main.py`** — terminal REPL. Runs a synchronous orchestrator loop.
5. **`agent.py`** — an older single-agent version (not multi-agent). Uses `tools/__init__.py` directly.

### Tool modules (`tools/`)

| Module | Agent | External APIs |
|---|---|---|
| `market_tools.py` | Market Analyst | yfinance |
| `news_tools.py` | News Analyst | Finnhub |
| `risk_tools.py` | Risk Manager | Alpaca, yfinance |
| `trading_tools.py` | Trader | Alpaca, yfinance |

### Trading guardrails (in `trading_tools.py`)

- Max 100 shares per order
- Buy orders capped at 20% of buying power
- Stop-loss requires existing position of sufficient size
- Risk Manager enforces: no single stock > 20% portfolio, no sector > 40%, keep 20% cash reserve

### Data

- `watchlist.json` — persisted watchlist managed by the Risk Manager agent
- Market data comes from the 40-stock universe defined in `market_tools.STOCK_UNIVERSE`

### Observability

All LLM and tool calls are traced with Opik (`@track` decorator and `track_openai` wrapper). Traces viewable at comet.com/opik.

### Duplicate orchestrator prompts

The `ORCHESTRATOR_PROMPT` is duplicated verbatim in both `main.py` and `app.py`. The `_get_alpaca()` helper is duplicated in `risk_tools.py` and `trading_tools.py`. Keep them in sync when editing.
