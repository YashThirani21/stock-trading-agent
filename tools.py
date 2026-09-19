"""
Tools that the LLM agent can call.

- Market data (prices, history, indicators): yfinance (free, no key needed)
- Portfolio & trading: Alpaca paper trading API (real orders on a sandbox)
"""

import json
import os
from datetime import datetime

import yfinance as yf
import alpaca_trade_api as tradeapi

# ── Alpaca client (paper trading) ───────────────────────────────────────

ALPACA_BASE_URL = "https://paper-api.alpaca.markets"  # paper = sandbox, live = https://api.alpaca.markets
def get_alpaca_client():
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file")
    return tradeapi.REST(api_key, secret_key, ALPACA_BASE_URL, api_version="v2")


# ── Market data tools (yfinance — unchanged) ────────────────────────────

def get_stock_price(ticker: str) -> str:
    """Fetch the current price and basic info for a stock."""
    stock = yf.Ticker(ticker.upper())
    info = stock.info
    fast_info = stock.fast_info

    price = fast_info.get("lastPrice") or info.get("currentPrice") or info.get("regularMarketPrice")
    if price is None:
        return json.dumps({"error": f"Could not fetch price for {ticker}"})

    return json.dumps({
        "ticker": ticker.upper(),
        "price": round(price, 2),
        "currency": info.get("currency", "USD"),
        "name": info.get("shortName", ticker.upper()),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    })


def get_historical_data(ticker: str, period: str = "1mo") -> str:
    """Fetch historical OHLCV data. period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y."""
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period=period)

    if hist.empty:
        return json.dumps({"error": f"No historical data for {ticker}"})

    records = []
    for date, row in hist.tail(20).iterrows():
        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
        })

    return json.dumps({"ticker": ticker.upper(), "period": period, "data": records})


def calculate_indicators(ticker: str) -> str:
    """Calculate RSI, SMA-20, SMA-50, and MACD for a stock."""
    stock = yf.Ticker(ticker.upper())
    hist = stock.history(period="3mo")

    if len(hist) < 50:
        return json.dumps({"error": f"Not enough data to calculate indicators for {ticker}"})

    close = hist["Close"]

    sma_20 = round(close.rolling(20).mean().iloc[-1], 2)
    sma_50 = round(close.rolling(50).mean().iloc[-1], 2)

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = round((100 - (100 / (1 + rs))).iloc[-1], 2)

    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9).mean()

    current_price = round(close.iloc[-1], 2)

    return json.dumps({
        "ticker": ticker.upper(),
        "price": current_price,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "rsi_14": rsi,
        "macd": round(macd_line.iloc[-1], 2),
        "macd_signal": round(signal_line.iloc[-1], 2),
        "macd_histogram": round((macd_line - signal_line).iloc[-1], 2),
        "trend": "BULLISH" if sma_20 > sma_50 else "BEARISH",
    })


# ── Portfolio & trading tools (now using Alpaca) ────────────────────────

def get_portfolio() -> str:
    """Return current portfolio from Alpaca: cash, holdings, P&L."""
    try:
        api = get_alpaca_client()
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


def place_order(ticker: str, qty: int, side: str) -> str:
    """
    Place a real paper trade via Alpaca.
    side: "buy" or "sell"
    Order type: market order, good-for-day.
    """
    ticker = ticker.upper()
    side = side.lower()

    if side not in ("buy", "sell"):
        return json.dumps({"error": "side must be 'buy' or 'sell'"})
    if qty <= 0:
        return json.dumps({"error": "qty must be positive"})

    # ── Guardrail: max 100 shares per order ──
    if qty > 100:
        return json.dumps({"error": "Guardrail: max 100 shares per single order. Reduce qty."})

    try:
        api = get_alpaca_client()

        # ── Guardrail: don't spend more than 20% of buying power on one buy ──
        if side == "buy":
            account = api.get_account()
            buying_power = float(account.buying_power)
            stock = yf.Ticker(ticker)
            price = stock.fast_info.get("lastPrice", 0)
            estimated_cost = price * qty
            if estimated_cost > buying_power * 0.20:
                return json.dumps({
                    "error": f"Guardrail: estimated cost ${estimated_cost:.2f} exceeds 20% of buying power (${buying_power * 0.20:.2f}). Reduce size."
                })

        order = api.submit_order(
            symbol=ticker,
            qty=qty,
            side=side,
            type="market",
            time_in_force="day",
        )

        return json.dumps({
            "status": order.status,
            "order_id": order.id,
            "ticker": order.symbol,
            "side": order.side,
            "qty": int(order.qty),
            "type": order.type,
            "submitted_at": str(order.submitted_at),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_order_status(order_id: str) -> str:
    """Check the status of a previously placed order."""
    try:
        api = get_alpaca_client()
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


# ── Tool registry ───────────────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "get_stock_price": get_stock_price,
    "get_historical_data": get_historical_data,
    "calculate_indicators": calculate_indicators,
    "get_portfolio": get_portfolio,
    "place_order": place_order,
    "get_order_status": get_order_status,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price and basic info (PE ratio, market cap, 52-week range) for a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. AAPL"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_data",
            "description": "Fetch historical OHLCV (open/high/low/close/volume) price data for a stock. Returns up to 20 most recent data points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "period": {
                        "type": "string",
                        "description": "Time period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 5y",
                        "default": "1mo",
                    },
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_indicators",
            "description": "Calculate technical indicators for a stock: RSI (14), SMA (20 & 50), MACD, and trend direction. Use this to analyze whether a stock is overbought/oversold or trending up/down.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "View current Alpaca paper trading portfolio: cash balance, buying power, stock holdings with unrealized P&L, and total portfolio value.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_order",
            "description": "Place a paper trade via Alpaca (market order, good-for-day). Guardrails: max 100 shares per order, max 20% of buying power per buy. Always check indicators and portfolio before trading.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                    "qty": {"type": "integer", "description": "Number of shares"},
                    "side": {"type": "string", "enum": ["buy", "sell"], "description": "buy or sell"},
                },
                "required": ["ticker", "qty", "side"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Check the status of a previously placed order by its order ID. Shows whether it was filled, the fill price, and timestamps.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "The Alpaca order ID returned when the order was placed"}
                },
                "required": ["order_id"],
            },
        },
    },
]
