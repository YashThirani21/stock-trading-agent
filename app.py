"""
Chainlit web UI for the Multi-Agent Stock Trading System.

Run with:  chainlit run app.py

Features: expandable agent steps, live portfolio sidebar, trade history.
"""

import json

from dotenv import load_dotenv
load_dotenv()

import opik
import chainlit as cl

from orchestrator import create_client, create_messages, run_orchestrator
from tools.risk_tools import get_portfolio
from tools.trading_tools import _get_alpaca


# ── Sidebar: Portfolio + Trade History ──────────────────────────────────

async def update_sidebar():
    portfolio_data = json.loads(get_portfolio())

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

    try:
        api = _get_alpaca()
        orders = api.list_orders(status="all", limit=10, direction="desc")
    except Exception:
        orders = []

    if orders:
        history_lines = ["### Trade History", ""]
        for order in orders:
            qty = int(float(order.qty))
            if order.status == "filled" and order.filled_avg_price:
                info = f"filled @ ${float(order.filled_avg_price):.2f}"
            else:
                info = order.status
            submitted = str(order.submitted_at)[:16].replace("T", " ")
            history_lines.append(
                f"**{order.side.upper()}** {qty}x {order.symbol} "
                f"— {info} ({submitted})"
            )
        history_text = "\n".join(history_lines)
    else:
        history_text = "*No trades yet*"

    portfolio_element = cl.Text(name="Portfolio", content=portfolio_text, display="side")
    trades_element = cl.Text(name="Trade History", content=history_text, display="side")

    await cl.ElementSidebar.set_title("Portfolio")
    await cl.ElementSidebar.set_elements([portfolio_element, trades_element])


# ── Chat Lifecycle ──────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    cl.user_session.set("messages", create_messages())
    cl.user_session.set("client", create_client())

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

    msg = cl.Message(content="")
    await msg.send()

    response = await run_orchestrator(
        client, messages,
        on_token=msg.stream_token,
    )

    await msg.update()
    await update_sidebar()
