# Stock Trading Agent

Multi-agent stock trading system powered by GPT-4o-mini.

## Overview

A multi-agent system where an orchestrator coordinates four specialist agents via OpenAI tool-calling. Uses Alpaca paper trading for order execution, yfinance for real-time market data, and Finnhub for financial news and sentiment.

## Architecture

```
User → Orchestrator → Market Analyst  (yfinance market data)
                    → News Analyst    (Finnhub news & sentiment)
                    → Risk Manager    (portfolio risk analysis)
                    → Trader          (Alpaca order execution)
```

The orchestrator receives user queries and delegates to specialist agents via tool-calling. Each specialist runs its own ReAct loop (`agent_loop.py`) with dedicated tools. The orchestrator synthesizes their outputs into a final response.

## Running the System

There are two ways to use the system — same orchestrator, different interface:

### Web UI (recommended)

```bash
chainlit run app.py
```

Opens a browser-based chat at `http://localhost:8000` with:
- Expandable agent steps (see what each specialist did)
- Live portfolio sidebar (holdings, cash, P&L)
- Trade history
- **Stateful specialists** — agents remember prior interactions within the session

Use this for regular interaction with the trading system.

### Terminal REPL

```bash
python main.py
```

A lightweight text-only interface — no browser needed. Useful for:
- Quick testing or debugging
- Environments without a display
- Simpler output (just text, no UI chrome)

Note: specialists are **stateless** in this mode (fresh context each call, no Chainlit session).

### Docker

```bash
cp .env.example .env
# Fill in your API keys in .env

docker compose up
# Open http://localhost:8000
```

## Trading Guardrails

- Max 100 shares per order
- Buy orders capped at 20% of buying power
- Stop-loss requires existing position of sufficient size
- Risk Manager enforces: no single stock >20% of portfolio, no sector >40%, keep 20% cash reserve

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM | GPT-4o-mini (OpenAI) | Agent reasoning and decisions |
| Broker | Alpaca | Paper trading execution |
| Market Data | yfinance | Stock prices, history, fundamentals |
| News | Finnhub | Financial news and sentiment |
| Web UI | Chainlit | Interactive chat interface |
| Observability | Opik | LLM and tool call tracing |

## Prerequisites

- Python 3.11+
- Docker (optional)
- API keys from: [OpenAI](https://platform.openai.com/), [Alpaca](https://alpaca.markets/) (paper trading), [Finnhub](https://finnhub.io/), [Opik](https://www.comet.com/site/products/opik/)

## Quick Start

```bash
git clone <repo-url>
cd stock-trading-agent

python -m venv .venv
source .venv/bin/activate

pip install .

cp .env.example .env
# Fill in your API keys in .env

# Web UI (recommended)
chainlit run app.py

# Or terminal REPL
python main.py
```

## Configuration

All prompts and model settings live in `config.yaml`. Edit it to change the model, tune agent behavior, or adjust turn limits — no Python changes needed.

## Observability

All LLM and tool calls are traced with [Opik](https://www.comet.com/site/products/opik/). View traces at [comet.com/opik](https://www.comet.com/opik).

## Project Structure

```
├── app.py              # Chainlit web UI (sidebar, trade history)
├── main.py             # Terminal REPL (lightweight, no browser)
├── orchestrator.py     # Shared orchestrator loop (used by both frontends)
├── agents.py           # Specialist agent definitions
├── agent_loop.py       # Generic ReAct agent loop
├── config.yaml         # All prompts and model settings
├── config.py           # Loads config.yaml
├── tools/
│   ├── market_tools.py # Market data via yfinance
│   ├── news_tools.py   # News via Finnhub
│   ├── risk_tools.py   # Portfolio risk analysis
│   └── trading_tools.py# Order execution via Alpaca
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```
