"""
Chainlit web UI for the Multi-Agent Stock Trading System.

Run with:  chainlit run app.py

Features: expandable agent steps, live portfolio sidebar, trade history.
"""

import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import opik
import chainlit as cl

from orchestrator import create_client, create_messages, run_orchestrator
from tools.risk_tools import get_portfolio


# ── Sidebar: Portfolio + Trade History ──────────────────────────────────

async def update_sidebar():
    portfolio_data = json.loads(get_portfolio())
    trade_log = cl.user_session.get("trade_log") or []

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

    portfolio_element = cl.Text(name="Portfolio", content=portfolio_text, display="side")
    trades_element = cl.Text(name="Trade History", content=history_text, display="side")

    await cl.ElementSidebar.set_title("Portfolio")
    await cl.ElementSidebar.set_elements([portfolio_element, trades_element])


# ── Trade logging callback ─────────────────────────────────────────────

def log_trade(fn_name, fn_args, result):
    if fn_name == "call_trader" and "error" not in result.lower():
        try:
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


# ── Chat Lifecycle ──────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    cl.user_session.set("messages", create_messages())
    cl.user_session.set("client", create_client())
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


@cl.on_message
async def handle_message(message: cl.Message):
    messages = cl.user_session.get("messages")
    client = cl.user_session.get("client")

    messages.append({"role": "user", "content": message.content})

    response = await run_orchestrator(client, messages, on_agent_call=log_trade)

    await cl.Message(content=response).send()
    await update_sidebar()
