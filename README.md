# Stock Trading Agent

Multi-agent stock trading system powered by GPT-4o-mini.

## Overview

A multi-agent system where an orchestrator coordinates four specialist agents via OpenAI tool-calling. Uses Alpaca paper trading for order execution, yfinance for real-time market data, and Finnhub for financial news and sentiment. Includes two frontends: a Chainlit web UI and a terminal REPL.

## Architecture

```
User → Orchestrator → Market Analyst  (yfinance market data)
                    → News Analyst    (Finnhub news & sentiment)
                    → Risk Manager    (portfolio risk analysis)
                    → Trader          (Alpaca order execution)
```

The orchestrator receives user queries and delegates to specialist agents via tool-calling. Each specialist runs its own ReAct loop (`agent_loop.py`) with dedicated tools. The orchestrator synthesizes their outputs into a final response.

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

## Quick Start — Local

```bash
git clone <repo-url>
cd stock-trading-agent

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys in .env

# Web UI
chainlit run app.py

# Or terminal REPL
python main.py
```

## Quick Start — Docker

```bash
cp .env.example .env
# Fill in your API keys in .env

docker compose up
# Open http://localhost:8000
```

## Observability

All LLM and tool calls are traced with [Opik](https://www.comet.com/site/products/opik/). View traces at [comet.com/opik](https://www.comet.com/opik).

## Project Structure

```
├── app.py              # Chainlit web UI frontend
├── main.py             # Terminal REPL frontend
├── agents.py           # Specialist agent definitions and orchestrator tools
├── agent_loop.py       # Generic ReAct agent loop
├── tools/
│   ├── market_tools.py # Market data via yfinance
│   ├── news_tools.py   # News via Finnhub
│   ├── risk_tools.py   # Portfolio risk analysis
│   └── trading_tools.py# Order execution via Alpaca
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```
