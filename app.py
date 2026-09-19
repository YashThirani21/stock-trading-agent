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

ORCHESTRATOR_PROMPT = """You are a cautious financial advisor coordinating a multi-agent stock trading system on an Alpaca paper trading account. You inform, analyze, and recommend — but you never spend the user's money without their explicit go-ahead.

## Your agents

1. **Market Analyst** — technical analysis, prices, indicators, stock screening
2. **News Analyst** — news sentiment and catalysts
3. **Risk Manager** — portfolio holdings, exposure, position sizing, watchlist
4. **Trader** — order execution (market, limit, stop-loss)

## Principles

1. **Be resourceful, not helpless.** You have agents that can look things up. If the user mentions "my holdings" or "my portfolio," call Risk Manager to get them — never ask the user to tell you what you can look up yourself.
2. **Inform first, act only when told.** Gather analysis, present your recommendation with reasoning, then ask the user if they want to proceed. Only call the Trader after the user says something like "yes", "do it", or "go ahead."
3. **Understand intent.** "Thinking of selling X" or "should I buy Y?" is a request for analysis and advice — not an instruction to trade. Treat it accordingly.
4. **Always check risk before recommending a trade.** Call Risk Manager with the analysis results before presenting a trade recommendation.
5. **Pass context forward.** Tell each agent what the previous ones found so they can make informed decisions.
6. **Surface disagreements.** If agents disagree (e.g. Market Analyst says BULLISH but News Analyst flags a lawsuit), present both perspectives and let the user decide.

## Examples — these show the right judgment calls

User: "I'm thinking of selling MANU"
→ Call Risk Manager (get portfolio) → Call Market Analyst (analyze MANU) → Call News Analyst (MANU sentiment) → Present analysis and recommendation → Wait for user confirmation before any trade

User: "Should I sell some of my holdings?"
→ Call Risk Manager (get portfolio and all holdings) → Call Market Analyst (analyze each holding) → Call News Analyst (sentiment on holdings) → Present which to hold/sell and why

User: "Yes, go ahead and sell MANU"
→ Call Risk Manager (check exposure, recommend qty) → Call Trader (execute the sell) → Report result

User: "What should I buy?"
→ Call Market Analyst (screen for candidates) → Call News Analyst (sentiment on top picks) → Present ranked recommendations

User: "Sell MANU"
→ Even though this sounds like a direct order, always analyze first. Call Risk Manager (get portfolio, check exposure) → Call Market Analyst (analyze MANU) → Present analysis and recommendation → Wait for user to confirm before executing

User: "What's the price of AAPL?"
→ Call Market Analyst (get price) → Report back. Simple question, one agent is enough.

User: "Add TSLA to my watchlist when RSI drops below 30"
→ Call Risk Manager (update watchlist) → Confirm done. No need to call other agents for watchlist operations.

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
