"""Trading execution tools — assigned to the Trader agent."""

import json
import os

import yfinance as yf
import alpaca_trade_api as tradeapi

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"


def _get_alpaca():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")
    return tradeapi.REST(api_key, secret_key, ALPACA_BASE_URL, api_version="v2")


def place_order(ticker: str, qty: int, side: str) -> str:
    """Place a market order via Alpaca."""
    ticker = ticker.upper()
    side = side.lower()

    if side not in ("buy", "sell"):
        return json.dumps({"error": "side must be 'buy' or 'sell'"})
    if qty <= 0:
        return json.dumps({"error": "qty must be positive"})
    if qty > 100:
        return json.dumps({"error": "Guardrail: max 100 shares per single order."})

    try:
        api = _get_alpaca()

        if side == "buy":
            account = api.get_account()
            buying_power = float(account.buying_power)
            stock = yf.Ticker(ticker)
            price = stock.fast_info.get("lastPrice", 0)
            estimated_cost = price * qty
            if estimated_cost > buying_power * 0.20:
                return json.dumps({
                    "error": f"Guardrail: estimated cost ${estimated_cost:.2f} exceeds 20% of buying power (${buying_power * 0.20:.2f})."
                })

        order = api.submit_order(
            symbol=ticker, qty=qty, side=side, type="market", time_in_force="day",
        )

        return json.dumps({
            "status": order.status,
            "order_id": order.id,
            "ticker": order.symbol,
            "side": order.side,
            "qty": int(order.qty),
            "type": "market",
            "submitted_at": str(order.submitted_at),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def place_limit_order(ticker: str, qty: int, side: str, limit_price: float) -> str:
    """Place a limit order — only executes at the specified price or better."""
    ticker = ticker.upper()
    side = side.lower()

    if side not in ("buy", "sell"):
        return json.dumps({"error": "side must be 'buy' or 'sell'"})
    if qty <= 0 or qty > 100:
        return json.dumps({"error": "qty must be between 1 and 100"})
    if limit_price <= 0:
        return json.dumps({"error": "limit_price must be positive"})

    try:
        api = _get_alpaca()
        order = api.submit_order(
            symbol=ticker, qty=qty, side=side, type="limit",
            time_in_force="gtc", limit_price=str(limit_price),
        )

        return json.dumps({
            "status": order.status,
            "order_id": order.id,
            "ticker": order.symbol,
            "side": order.side,
            "qty": int(order.qty),
            "type": "limit",
            "limit_price": limit_price,
            "submitted_at": str(order.submitted_at),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def place_stop_loss(ticker: str, qty: int, stop_price: float) -> str:
    """Place a stop-loss sell order — triggers a market sell when price drops to stop_price."""
    ticker = ticker.upper()

    if qty <= 0 or qty > 100:
        return json.dumps({"error": "qty must be between 1 and 100"})
    if stop_price <= 0:
        return json.dumps({"error": "stop_price must be positive"})

    try:
        api = _get_alpaca()

        positions = {p.symbol: int(p.qty) for p in api.list_positions()}
        if ticker not in positions or positions[ticker] < qty:
            return json.dumps({
                "error": f"Cannot set stop-loss for {qty} shares of {ticker}. You hold {positions.get(ticker, 0)}."
            })

        order = api.submit_order(
            symbol=ticker, qty=qty, side="sell", type="stop",
            time_in_force="gtc", stop_price=str(stop_price),
        )

        return json.dumps({
            "status": order.status,
            "order_id": order.id,
            "ticker": order.symbol,
            "type": "stop_loss",
            "qty": int(order.qty),
            "stop_price": stop_price,
            "submitted_at": str(order.submitted_at),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_order_status(order_id: str) -> str:
    """Check the status of a previously placed order."""
    try:
        api = _get_alpaca()
        order = api.get_order(order_id)

        result = {
            "order_id": order.id,
            "ticker": order.symbol,
            "side": order.side,
            "qty": int(order.qty),
            "status": order.status,
            "type": order.type,
            "submitted_at": str(order.submitted_at),
        }
        if order.filled_at:
            result["filled_at"] = str(order.filled_at)
            result["filled_price"] = float(order.filled_avg_price)

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


FUNCTIONS = {
    "place_order": place_order,
    "place_limit_order": place_limit_order,
    "place_stop_loss": place_stop_loss,
    "get_order_status": get_order_status,
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place a market order (buy or sell). Submit anytime — fills at market price during hours, queued outside hours. Guardrails: max 100 shares, max 20% of buying power per buy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "qty": {"type": "integer", "description": "Number of shares"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                },
                "required": ["ticker", "qty", "side"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_limit_order",
            "description": "Place a limit order — only fills at the specified price or better. Good-till-cancelled. Use when you want to buy at a lower price or sell at a higher price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "qty": {"type": "integer", "description": "Number of shares"},
                    "side": {"type": "string", "enum": ["buy", "sell"]},
                    "limit_price": {"type": "number", "description": "Maximum price for buy, minimum price for sell"},
                },
                "required": ["ticker", "qty", "side", "limit_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_stop_loss",
            "description": "Place a stop-loss sell order on an existing position. Automatically sells if the price drops to the stop price. Protects against large losses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "qty": {"type": "integer", "description": "Number of shares to sell at stop"},
                    "stop_price": {"type": "number", "description": "Price at which to trigger the sell"},
                },
                "required": ["ticker", "qty", "stop_price"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Check the status of a previously placed order by order ID. Shows fill status, price, and timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The Alpaca order ID"}
                },
                "required": ["order_id"],
            },
        },
    },
]
