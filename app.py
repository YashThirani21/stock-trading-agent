"""
Chainlit frontend for the Multi-Agent Stock Trading System.

Run with:  chainlit run app.py

This replaces main.py's terminal REPL with a browser-based chat UI.
The orchestrator logic is the same — specialist agents appear as
expandable steps in the chat.

Sidebar shows: live portfolio + trade history (updates after every message).
"""

import json
import os
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import opik

import chainlit as cl
from openai import OpenAI
from opik import track
from opik.integrations.openai import track_openai
from agents import AGENT_SCHEMAS, AGENT_FUNCTIONS
from tools.risk_tools import get_portfolio

MODEL = "gpt-4o-mini"
MAX_TURNS = 15

ORCHESTRATOR_PROMPT = """You are the lead coordinator of a multi-agent stock trading system connected to an Alpaca paper trading account.

You have 4 specialist agents you can delegate to:

1. **Market Analyst** — technical analysis (prices, indicators, trends) AND stock screening/recommendations
2. **News Analyst** — news sentiment and catalysts
3. **Risk Manager** — portfolio exposure, position sizing, watchlist management
4. **Trader** — order execution (market, limit, stop-loss orders)

Your workflow for trade decisions:
1. FIRST: Call Market Analyst for technical analysis
2. THEN: Call News Analyst for sentiment check
3. THEN: Call Risk Manager with both analyses — ask if the trade is safe and what size
4. STOP and present your recommendation to the user: what to trade, why, and the recommended qty/price. Ask for their explicit confirmation before proceeding.
5. ONLY after the user explicitly confirms (e.g. "yes", "go ahead", "do it"): Call Trader to execute
6. Summarize the execution result

CRITICAL — User confirmation rules:
- NEVER call the Trader unless the user has explicitly confirmed the trade in their message
- Vague or exploratory language ("thinking of selling", "should I sell?", "considering buying") is NOT confirmation — treat it as a request for analysis and recommendation only
- The user must give a clear, affirmative instruction to execute (e.g. "yes sell it", "buy 10 shares", "go ahead")
- If in doubt whether the user wants to execute, ask — do NOT assume

Your workflow for stock recommendations ("what should I buy?"):
1. Call Market Analyst to screen for candidates (e.g. bullish stocks, oversold opportunities)
2. Call News Analyst for sentiment on the top picks
3. Present the ranked recommendations to the user

Rules:
- NEVER skip the Risk Manager before trading
- NEVER execute a trade without explicit user confirmation — analysis and risk approval alone are not enough
- Pass relevant context between agents (e.g., tell Risk Manager what the Analyst found)
- If agents disagree, explain the conflict and let the user decide
- For simple questions (price check, portfolio view), you can call just one agent
- For watchlist operations, delegate directly to Risk Manager
- Always give the user a clear summary of what happened and why

You coordinate — you don't analyze or trade directly."""


# ── Sidebar: Portfolio + Trade History ──────────────────────────────────

async def update_sidebar():
    """Fetch portfolio from Alpaca and render it in the sidebar."""
    portfolio_data = json.loads(get_portfolio())
    trade_log = cl.user_session.get("trade_log") or []

    # Build portfolio text
    if "error" in portfolio_data:
        portfolio_text = f"**Error:** {portfolio_data['error']}"
    else:
        lines = [
            f"**Cash:** ${portfolio_data['cash']:,.2f}",
            f"**Portfolio Value:** ${portfolio_data['portfolio_value']:,.2f}",
            f"**Buying Power:** ${portfolio_data['buying_power']:,.2f}",
            "",
        ]

        holdings = portfolio_data.get("holdings", {})
        if holdings:
            lines.append("### Holdings")
            lines.append("")
            for ticker, pos in holdings.items():
                pnl = pos['unrealized_pnl']
                pnl_sign = "+" if pnl >= 0 else ""
                lines.append(
                    f"**{ticker}** — {pos['qty']} shares @ ${pos['avg_price']:.2f}\n"
                    f"Now: ${pos['current_price']:.2f} | "
                    f"P&L: {pnl_sign}${pnl:.2f} ({pos['unrealized_pnl_pct']:+.1f}%)"
                )
                lines.append("")
        else:
            lines.append("*No holdings*")

        portfolio_text = "\n".join(lines)

    # Build trade history text
    if trade_log:
        history_lines = ["### Trade History", ""]
        for trade in reversed(trade_log[-10:]):
            history_lines.append(
                f"**{trade['side'].upper()}** {trade['qty']}x {trade['ticker']} "
                f"@ {trade['time']}"
            )
        history_text = "\n".join(history_lines)
    else:
        history_text = "*No trades this session*"

    portfolio_element = cl.Text(
        name="Portfolio",
        content=portfolio_text,
        display="side",
    )
    trades_element = cl.Text(
        name="Trade History",
        content=history_text,
        display="side",
    )

    await cl.ElementSidebar.set_title("Portfolio")
    await cl.ElementSidebar.set_elements([portfolio_element, trades_element])


# ── Chat Lifecycle ──────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    messages = [{"role": "system", "content": ORCHESTRATOR_PROMPT}]
    cl.user_session.set("messages", messages)
    cl.user_session.set("client", track_openai(OpenAI()))
    cl.user_session.set("trade_log", [])

    await update_sidebar()

    await cl.Message(
        content="Welcome to the **Multi-Agent Stock Trading System**.\n\n"
        "I coordinate 4 specialist agents:\n"
        "- **Market Analyst** — prices, indicators, stock screening\n"
        "- **News Analyst** — sentiment and catalysts\n"
        "- **Risk Manager** — portfolio exposure and position sizing\n"
        "- **Trader** — order execution via Alpaca\n\n"
        "Try: *\"What's the price of AAPL?\"* or *\"Should I buy TSLA?\"* or *\"Show my portfolio\"*\n\n"
        "Your portfolio is shown in the sidebar."
    ).send()


# ── Orchestrator ────────────────────────────────────────────────────────

@track(name="orchestrator", capture_input=True, capture_output=True)
async def run_orchestrator(client, messages):
    for turn in range(MAX_TURNS):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=AGENT_SCHEMAS,
            tool_choice="auto",
        )

        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump())

        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        for tool_call in assistant_message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name in AGENT_FUNCTIONS:
                result = await AGENT_FUNCTIONS[fn_name](**fn_args)
            else:
                result = f"Unknown agent: {fn_name}"

            # Track trades in session for the sidebar
            if fn_name == "call_trader" and "error" not in result.lower():
                try:
                    # Try to extract trade info from the trader's response
                    trade_log = cl.user_session.get("trade_log") or []
                    trade_log.append({
                        "ticker": fn_args.get("query", ""),
                        "qty": "",
                        "side": "",
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "detail": result[:100],
                    })
                    cl.user_session.set("trade_log", trade_log)
                except Exception:
                    pass

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Orchestrator reached max turns."


@cl.on_message
async def handle_message(message: cl.Message):
    messages = cl.user_session.get("messages")
    client = cl.user_session.get("client")

    messages.append({"role": "user", "content": message.content})

    response = await run_orchestrator(client, messages)

    await cl.Message(content=response).send()

    # Refresh the sidebar after every message (portfolio may have changed)
    await update_sidebar()
