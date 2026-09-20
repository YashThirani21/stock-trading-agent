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

import re

import chainlit as cl
from pydantic import ValidationError

from agent_loop import run_agent, _has_chainlit_context
from config import (
    MARKET_ANALYST_PROMPT, NEWS_ANALYST_PROMPT,
    RISK_MANAGER_PROMPT, TRADER_PROMPT,
)
from schemas import TradeOrder
from tools import market_tools, news_tools, risk_tools, trading_tools


_TRADE_EVAL_PATTERN = re.compile(
    r'\b(buy|sell|trade|safe|approve|position siz)', re.IGNORECASE
)


def _is_trade_eval(query: str) -> bool:
    if re.search(r'\bwatchlist\b', query, re.IGNORECASE):
        return False
    return bool(_TRADE_EVAL_PATTERN.search(query))


RISK_APPROVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["APPROVE", "REJECT"]},
        "recommended_qty": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "reason": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decision", "recommended_qty", "reason", "warnings"],
    "additionalProperties": False,
}


def _get_agent_messages(agent_name: str, system_prompt: str) -> list | None:
    if not _has_chainlit_context():
        return None
    key = f"agent_messages_{agent_name}"
    messages = cl.user_session.get(key)
    if messages is None:
        messages = [{"role": "system", "content": system_prompt}]
        cl.user_session.set(key, messages)
    return messages


# ── Market Analyst ──────────────────────────────────────────────────────

async def call_market_analyst(query: str) -> str:
    messages = _get_agent_messages("market_analyst", MARKET_ANALYST_PROMPT)
    if _has_chainlit_context():
        async with cl.Step(name="Market Analyst", type="tool") as step:
            step.input = query
            await step.send()
            result = await run_agent(
                MARKET_ANALYST_PROMPT, query,
                market_tools.SCHEMAS, market_tools.FUNCTIONS,
                messages=messages,
                parent_step=step,
            )
            step.output = result
    else:
        result = await run_agent(
            MARKET_ANALYST_PROMPT, query,
            market_tools.SCHEMAS, market_tools.FUNCTIONS,
            messages=messages,
        )
    return result


# ── News Analyst ────────────────────────────────────────────────────────

async def call_news_analyst(query: str) -> str:
    messages = _get_agent_messages("news_analyst", NEWS_ANALYST_PROMPT)
    if _has_chainlit_context():
        async with cl.Step(name="News Analyst", type="tool") as step:
            step.input = query
            await step.send()
            result = await run_agent(
                NEWS_ANALYST_PROMPT, query,
                news_tools.SCHEMAS, news_tools.FUNCTIONS,
                messages=messages,
                parent_step=step,
            )
            step.output = result
    else:
        result = await run_agent(
            NEWS_ANALYST_PROMPT, query,
            news_tools.SCHEMAS, news_tools.FUNCTIONS,
            messages=messages,
        )
    return result


# ── Risk Manager ────────────────────────────────────────────────────────

async def call_risk_manager(query: str) -> str:
    messages = _get_agent_messages("risk_manager", RISK_MANAGER_PROMPT)
    schema = RISK_APPROVAL_SCHEMA if _is_trade_eval(query) else None
    if _has_chainlit_context():
        async with cl.Step(name="Risk Manager", type="tool") as step:
            step.input = query
            await step.send()
            result = await run_agent(
                RISK_MANAGER_PROMPT, query,
                risk_tools.SCHEMAS, risk_tools.FUNCTIONS,
                messages=messages,
                parent_step=step,
                response_schema=schema,
            )
            step.output = result
    else:
        result = await run_agent(
            RISK_MANAGER_PROMPT, query,
            risk_tools.SCHEMAS, risk_tools.FUNCTIONS,
            messages=messages,
            response_schema=schema,
        )
    return result


# ── Trader ──────────────────────────────────────────────────────────────

async def call_trader(
    ticker: str,
    qty: int,
    side: str,
    order_type: str = "market",
    limit_price: float | None = None,
    stop_price: float | None = None,
) -> str:
    try:
        order = TradeOrder(
            ticker=ticker, qty=qty, side=side, order_type=order_type,
            limit_price=limit_price, stop_price=stop_price,
        )
    except ValidationError as e:
        return f"Trade rejected — invalid order: {e.errors()}"

    if order.order_type == "market":
        instruction = f"Place a market order: {order.side} {order.qty} shares of {order.ticker}"
    elif order.order_type == "limit":
        instruction = f"Place a limit order: {order.side} {order.qty} shares of {order.ticker} at ${order.limit_price}"
    elif order.order_type == "stop_loss":
        instruction = f"Place a stop-loss order: sell {order.qty} shares of {order.ticker} at stop price ${order.stop_price}"

    messages = _get_agent_messages("trader", TRADER_PROMPT)
    step_input = order.model_dump_json(indent=2)
    if _has_chainlit_context():
        async with cl.Step(name="Trader", type="tool") as step:
            step.input = step_input
            await step.send()
            result = await run_agent(
                TRADER_PROMPT, instruction,
                trading_tools.SCHEMAS, trading_tools.FUNCTIONS,
                messages=messages,
                parent_step=step,
            )
            step.output = result
    else:
        result = await run_agent(
            TRADER_PROMPT, instruction,
            trading_tools.SCHEMAS, trading_tools.FUNCTIONS,
            messages=messages,
        )
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
            "description": "Execute a trade that has been approved by the user and risk-checked. Provide structured order details. ONLY call after user explicitly confirms.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g. TSLA, AAPL)"},
                    "qty": {"type": "integer", "description": "Number of shares (1-100)"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "order_type": {"type": "string", "enum": ["market", "limit", "stop_loss"], "default": "market"},
                    "limit_price": {"type": "number", "description": "Required for limit orders — the price to buy/sell at"},
                    "stop_price": {"type": "number", "description": "Required for stop_loss orders — the trigger price"},
                },
                "required": ["ticker", "qty", "side"],
            },
        },
    },
]
