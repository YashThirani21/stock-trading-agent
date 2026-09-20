"""Risk management tools — assigned to the Risk Manager agent."""

import json
import os
from pathlib import Path

import yfinance as yf
import alpaca_trade_api as tradeapi

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
WATCHLIST_PATH = Path(__file__).parent.parent / "watchlist.json"


def _get_alpaca():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")
    return tradeapi.REST(api_key, secret_key, ALPACA_BASE_URL, api_version="v2")


def get_portfolio() -> str:
    """Return current portfolio from Alpaca: cash, holdings, P&L."""
    try:
        api = _get_alpaca()
        account = api.get_account()
        positions = api.list_positions()

        holdings = {}
        for pos in positions:
            holdings[pos.symbol] = {
                "qty": int(pos.qty),
                "avg_price": round(float(pos.avg_entry_price), 2),
                "current_price": round(float(pos.current_price), 2),
                "market_value": round(float(pos.market_value), 2),
                "unrealized_pnl": round(float(pos.unrealized_pl), 2),
                "unrealized_pnl_pct": round(float(pos.unrealized_plpc) * 100, 2),
            }

        return json.dumps({
            "cash": round(float(account.cash), 2),
            "portfolio_value": round(float(account.portfolio_value), 2),
            "buying_power": round(float(account.buying_power), 2),
            "holdings": holdings,
            "account_status": account.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def check_exposure() -> str:
    """Analyze portfolio concentration by stock and sector."""
    try:
        api = _get_alpaca()
        account = api.get_account()
        positions = api.list_positions()
        portfolio_value = float(account.portfolio_value)

        if not positions:
            return json.dumps({"message": "No positions — portfolio is 100% cash", "cash_pct": 100.0})

        stock_exposure = {}
        sector_exposure = {}

        for pos in positions:
            market_value = float(pos.market_value)
            pct = round((market_value / portfolio_value) * 100, 2)

            stock = yf.Ticker(pos.symbol)
            sector = stock.info.get("sector", "Unknown")

            stock_exposure[pos.symbol] = {"pct_of_portfolio": pct, "sector": sector}
            sector_exposure[sector] = round(sector_exposure.get(sector, 0) + pct, 2)

        cash_pct = round((float(account.cash) / portfolio_value) * 100, 2)

        warnings = []
        for ticker, data in stock_exposure.items():
            if data["pct_of_portfolio"] > 20:
                warnings.append(f"{ticker} is {data['pct_of_portfolio']}% of portfolio (over 20% limit)")
        for sector, pct in sector_exposure.items():
            if pct > 40:
                warnings.append(f"{sector} sector is {pct}% of portfolio (over 40% limit)")

        return json.dumps({
            "stock_exposure": stock_exposure,
            "sector_exposure": sector_exposure,
            "cash_pct": cash_pct,
            "warnings": warnings,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _load_watchlist() -> dict:
    if WATCHLIST_PATH.exists():
        return json.loads(WATCHLIST_PATH.read_text())
    return {"entries": []}


def _save_watchlist(data: dict):
    WATCHLIST_PATH.write_text(json.dumps(data, indent=2))


def get_watchlist() -> str:
    """Return the current watchlist."""
    wl = _load_watchlist()
    return json.dumps(wl)


def update_watchlist(action: str, ticker: str, condition: str = "", threshold: float = 0, note: str = "") -> str:
    """
    Add or remove a watchlist entry.
    action: "add" or "remove"
    condition (for add): "rsi_below", "rsi_above", "price_below", "price_above"
    threshold (for add): the numeric threshold
    note (for add): optional reminder of why
    """
    note = note[:200]
    wl = _load_watchlist()

    if action == "remove":
        before = len(wl["entries"])
        wl["entries"] = [e for e in wl["entries"] if e["ticker"] != ticker.upper()]
        _save_watchlist(wl)
        removed = before - len(wl["entries"])
        return json.dumps({"status": "ok", "removed": removed, "remaining": len(wl["entries"])})

    if action == "add":
        if not condition:
            return json.dumps({"error": "condition is required for add (rsi_below, rsi_above, price_below, price_above)"})

        entry = {
            "ticker": ticker.upper(),
            "condition": condition,
            "threshold": threshold,
            "note": note,
        }
        wl["entries"] = [e for e in wl["entries"] if not (e["ticker"] == ticker.upper() and e["condition"] == condition)]
        wl["entries"].append(entry)
        _save_watchlist(wl)
        return json.dumps({"status": "added", "entry": entry, "total_entries": len(wl["entries"])})

    return json.dumps({"error": f"Unknown action: {action}. Use 'add' or 'remove'"})


FUNCTIONS = {
    "get_portfolio": get_portfolio,
    "check_exposure": check_exposure,
    "get_watchlist": get_watchlist,
    "update_watchlist": update_watchlist,
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "View current Alpaca paper trading portfolio: cash balance, buying power, stock holdings with unrealized P&L.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_exposure",
            "description": "Analyze portfolio concentration risk — shows what % of portfolio is in each stock and sector, and flags any positions over concentration limits (20% per stock, 40% per sector).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_watchlist",
            "description": "View the current watchlist of stocks being monitored with their alert conditions.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_watchlist",
            "description": "Add or remove a stock from the watchlist with alert conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["add", "remove"], "description": "add or remove"},
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "condition": {
                        "type": "string",
                        "enum": ["rsi_below", "rsi_above", "price_below", "price_above"],
                        "description": "Alert condition (required for add)",
                    },
                    "threshold": {"type": "number", "description": "Numeric threshold for the condition"},
                    "note": {"type": "string", "description": "Optional note explaining the alert"},
                },
                "required": ["action", "ticker"],
            },
        },
    },
]
