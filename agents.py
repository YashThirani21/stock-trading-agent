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
from config import (
    MARKET_ANALYST_PROMPT, NEWS_ANALYST_PROMPT,
    RISK_MANAGER_PROMPT, TRADER_PROMPT,
)
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
