"""
Specialist agent definitions.

Each agent is just a configuration: a system prompt + a set of tools.
They all share the same ReAct loop from agent_loop.py.

The orchestrator calls these as functions — each one runs a full agent loop
internally and returns the agent's text response.

In the Chainlit UI, each specialist shows as a named expandable step,
with its tool calls nested inside.

Stateful: each specialist's message history is stored in cl.user_session
so it remembers previous interactions within the same chat session.
"""

import chainlit as cl
from agent_loop import run_agent
from tools import market_tools, news_tools, risk_tools, trading_tools


def _get_agent_messages(agent_name: str, system_prompt: str) -> list:
    """Get or create a persistent message history for a specialist agent."""
    key = f"agent_messages_{agent_name}"
    messages = cl.user_session.get(key)
    if messages is None:
        messages = [{"role": "system", "content": system_prompt}]
        cl.user_session.set(key, messages)
    return messages


# ── Market Analyst ──────────────────────────────────────────────────────

MARKET_ANALYST_PROMPT = """You are a Market Analyst specialist. Your ONLY job is technical and fundamental analysis.

You have tools to:
- Fetch current stock prices and company info
- Retrieve historical price data
- Calculate technical indicators (RSI, SMA, MACD)
- Screen stocks to find buy/sell candidates from a universe of 40 major US stocks

When asked to analyze a specific stock:
1. Get the current price and basic info
2. Calculate technical indicators
3. Provide a clear summary: trend direction, overbought/oversold status, key support/resistance levels
4. Give a recommendation: BULLISH, BEARISH, or NEUTRAL with confidence level

When asked to find or recommend stocks:
1. Use the stock screener with appropriate filters (sector, signal, market cap)
2. For the top candidates, calculate detailed indicators
3. Rank them and explain why each is interesting

You do NOT have access to portfolio data or trading. Just analyze and report.
Keep your analysis concise — the orchestrator will pass it to other agents."""


async def call_market_analyst(query: str) -> str:
    async with cl.Step(name="Market Analyst", type="tool") as step:
        step.input = query
        messages = _get_agent_messages("market_analyst", MARKET_ANALYST_PROMPT)
        result = await run_agent(
            MARKET_ANALYST_PROMPT, query,
            market_tools.SCHEMAS, market_tools.FUNCTIONS,
            messages=messages,
        )
        step.output = result
    return result


# ── News Analyst ────────────────────────────────────────────────────────

NEWS_ANALYST_PROMPT = """You are a News Analyst specialist. Your ONLY job is analyzing news sentiment.

You have tools to:
- Fetch recent company-specific news headlines
- Fetch general market news

When asked about a stock or market:
1. Fetch relevant news
2. Analyze the sentiment of each headline: positive, negative, or neutral
3. Identify any major catalysts, risks, or events
4. Provide an overall sentiment score: POSITIVE, NEGATIVE, or MIXED
5. Flag any red flags (lawsuits, SEC investigations, earnings misses, executive departures)

You do NOT have access to price data or trading. Just analyze news and report sentiment.
Keep your analysis concise — the orchestrator will pass it to other agents."""


async def call_news_analyst(query: str) -> str:
    async with cl.Step(name="News Analyst", type="tool") as step:
        step.input = query
        messages = _get_agent_messages("news_analyst", NEWS_ANALYST_PROMPT)
        result = await run_agent(
            NEWS_ANALYST_PROMPT, query,
            news_tools.SCHEMAS, news_tools.FUNCTIONS,
            messages=messages,
        )
        step.output = result
    return result


# ── Risk Manager ────────────────────────────────────────────────────────

RISK_MANAGER_PROMPT = """You are a Risk Manager specialist. Your ONLY job is portfolio risk assessment and position sizing.

You have tools to:
- View the current Alpaca paper trading portfolio
- Analyze portfolio concentration (stock and sector exposure)
- View and manage the watchlist

When asked to evaluate a potential trade:
1. Check the current portfolio and exposure
2. Assess whether the trade would create concentration risk
3. Recommend a position size that keeps:
   - No single stock > 20% of portfolio
   - No single sector > 40% of portfolio
   - Always keep at least 20% cash reserve
4. Either APPROVE with a recommended qty, or REJECT with the reason

When asked about the watchlist, manage entries as requested.

You do NOT have access to price analysis or trade execution. Just assess risk and recommend sizing.
Keep your response concise — the orchestrator will pass it to other agents."""


async def call_risk_manager(query: str) -> str:
    async with cl.Step(name="Risk Manager", type="tool") as step:
        step.input = query
        messages = _get_agent_messages("risk_manager", RISK_MANAGER_PROMPT)
        result = await run_agent(
            RISK_MANAGER_PROMPT, query,
            risk_tools.SCHEMAS, risk_tools.FUNCTIONS,
            messages=messages,
        )
        step.output = result
    return result


# ── Trader ──────────────────────────────────────────────────────────────

TRADER_PROMPT = """You are a Trader specialist. Your ONLY job is executing trades on the Alpaca paper trading account.

You have tools to:
- Place market orders (immediate execution)
- Place limit orders (execute at a specific price or better)
- Place stop-loss orders (auto-sell if price drops below a threshold)
- Check order status

Rules:
1. ONLY execute trades when the orchestrator explicitly tells you to place an order with specific details (ticker, qty, side). The orchestrator is responsible for getting user confirmation first — but if the instructions you receive sound uncertain or exploratory, refuse and ask the orchestrator to confirm with the user.
2. If the orchestrator provides ticker, qty, and side — place the order directly without looking up prices. Those details have already been decided and confirmed.
3. If any required detail is missing or unclear (e.g. no qty specified), ask for clarification instead of guessing
4. After placing an order, check its status and report the result
5. If an order fails, report the error clearly
6. Market orders only fill during US market hours (9:30 AM - 4:00 PM ET, Mon-Fri)
7. Do NOT do position sizing or analysis — that's the Risk Manager's and Market Analyst's job

You do NOT analyze stocks or assess risk. You just execute approved trades.
Keep your response concise — report what was executed and the result."""


async def call_trader(query: str) -> str:
    async with cl.Step(name="Trader", type="tool") as step:
        step.input = query
        messages = _get_agent_messages("trader", TRADER_PROMPT)
        result = await run_agent(
            TRADER_PROMPT, query,
            trading_tools.SCHEMAS, trading_tools.FUNCTIONS,
            messages=messages,
        )
        step.output = result
    return result


# ── Registry: these become the orchestrator's "tools" ───────────────────

AGENT_FUNCTIONS = {
    "call_market_analyst": call_market_analyst,
    "call_news_analyst": call_news_analyst,
    "call_risk_manager": call_risk_manager,
    "call_trader": call_trader,
}

AGENT_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "call_market_analyst",
            "description": "Ask the Market Analyst to analyze a stock's technicals OR screen for stock recommendations. Use for price analysis, technical signals, AND 'what should I buy?' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to analyze, e.g. 'Analyze TSLA technicals' or 'Screen for oversold large-cap tech stocks'"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_news_analyst",
            "description": "Ask the News Analyst to assess recent news sentiment for a stock or the overall market. Use to check for catalysts, red flags, or sentiment before trading.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to research, e.g. 'What is the recent news sentiment for AAPL?'"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_risk_manager",
            "description": "Ask the Risk Manager to check portfolio exposure, assess whether a trade is safe, and recommend position sizing. Also manages the watchlist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to assess, e.g. 'Can we safely buy TSLA? Check exposure and recommend qty.' or 'Add MSFT to watchlist with RSI below 30'"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_trader",
            "description": "Ask the Trader to execute a specific trade that has been approved. Also checks order status. ONLY call after the Risk Manager has approved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to execute, e.g. 'Buy 15 shares of TSLA at market price' or 'Place a stop-loss on TSLA at $230 for 15 shares'"}
                },
                "required": ["query"],
            },
        },
    },
]
