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

import asyncio

import chainlit as cl
from pydantic import ValidationError

from agent_loop import run_agent, _has_chainlit_context
from config import (
    MARKET_ANALYST_PROMPT, NEWS_ANALYST_PROMPT,
    RISK_MANAGER_PROMPT,
)
from schemas import TradeOrder
import re

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
            ticker=ticker.upper(), qty=qty, side=side.lower(), order_type=order_type,
            limit_price=limit_price, stop_price=stop_price,
        )
    except ValidationError as e:
        return f"Trade rejected — invalid order: {e.errors()}"

    if order.order_type == "limit" and order.limit_price is not None:
        summary = f"{order.side.upper()} {order.qty} shares of {order.ticker} (limit order @ ${order.limit_price:.2f})"
    elif order.order_type == "stop_loss" and order.stop_price is not None:
        summary = f"SELL {order.qty} shares of {order.ticker} (stop-loss @ ${order.stop_price:.2f})"
    else:
        summary = f"{order.side.upper()} {order.qty} shares of {order.ticker} (market order)"

    if _has_chainlit_context():
        res = await cl.AskActionMessage(
            content=f"**Trade Confirmation**\n\n{summary}\n\nDo you want to proceed?",
            actions=[
                cl.Action(name="confirm", label="Confirm Trade", payload={"action": "confirm"}),
                cl.Action(name="cancel", label="Cancel", payload={"action": "cancel"}),
            ],
            timeout=120,
        ).send()
        if not res or res.get("name") != "confirm":
            return "Trade cancelled by user."
    else:
        confirm = await asyncio.to_thread(input, f"\n⚠️  {summary}\nConfirm? [y/n]: ")
        if confirm.strip().lower() not in ("y", "yes"):
            return "Trade cancelled by user."

    if order.order_type == "limit":
        execute = lambda: trading_tools.place_limit_order(
            order.ticker, order.qty, order.side, order.limit_price,
        )
    elif order.order_type == "stop_loss":
        execute = lambda: trading_tools.place_stop_loss(
            order.ticker, order.qty, order.stop_price,
        )
    else:
        execute = lambda: trading_tools.place_order(
            order.ticker, order.qty, order.side,
        )

    if _has_chainlit_context():
        async with cl.Step(name="Trader", type="tool") as step:
            step.input = summary
            await step.send()
            result = await asyncio.to_thread(execute)
            step.output = result
    else:
        result = await asyncio.to_thread(execute)

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
            "description": "Look up the user's current holdings and portfolio, check exposure and concentration risk, assess whether a trade is safe, or manage the watchlist. Call this first whenever you need to know what the user owns.",
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
