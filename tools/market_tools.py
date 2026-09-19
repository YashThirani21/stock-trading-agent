"""Market data tools — assigned to the Market Analyst agent."""

import json
import yfinance as yf

STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "MA", "HD", "PG", "JNJ", "MRK", "ABBV", "CVX", "XOM", "KO",
    "PEP", "COST", "WMT", "BAC", "CRM", "NFLX", "AMD", "INTC", "DIS", "CSCO",
    "ADBE", "CMCSA", "PFE", "NKE", "T", "VZ", "NEE", "LIN", "UNP", "LOW",
]


def get_stock_price(ticker: str) -> str:
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
        "sector": info.get("sector"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    })


def get_historical_data(ticker: str, period: str = "1mo") -> str:
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


def screen_stocks(sector: str = "", signal: str = "any", min_market_cap: str = "any") -> str:
    """Screen stocks from a universe of 40 major US stocks by sector, signal, and market cap."""
    results = []

    for ticker in STOCK_UNIVERSE:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fast_info = stock.fast_info

            stock_sector = info.get("sector", "Unknown")
            if sector and sector.lower() not in stock_sector.lower():
                continue

            mcap = info.get("marketCap", 0)
            if min_market_cap == "large" and mcap < 100_000_000_000:
                continue
            elif min_market_cap == "mid" and (mcap < 10_000_000_000 or mcap > 100_000_000_000):
                continue

            price = fast_info.get("lastPrice") or info.get("currentPrice")
            if not price:
                continue

            hist = stock.history(period="3mo")
            if len(hist) < 50:
                continue

            close = hist["Close"]
            sma_20 = close.rolling(20).mean().iloc[-1]
            sma_50 = close.rolling(50).mean().iloc[-1]

            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = round((100 - (100 / (1 + rs))).iloc[-1], 2)

            trend = "BULLISH" if sma_20 > sma_50 else "BEARISH"

            if signal == "oversold" and rsi > 30:
                continue
            elif signal == "overbought" and rsi < 70:
                continue
            elif signal == "bullish" and trend != "BULLISH":
                continue
            elif signal == "bearish" and trend != "BEARISH":
                continue

            results.append({
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "sector": stock_sector,
                "price": round(price, 2),
                "rsi_14": rsi,
                "trend": trend,
                "pe_ratio": info.get("trailingPE"),
                "market_cap_b": round(mcap / 1_000_000_000, 1) if mcap else None,
            })

            if len(results) >= 10:
                break
        except Exception:
            continue

    if not results:
        return json.dumps({"message": "No stocks matched your criteria. Try broader filters."})

    return json.dumps({"matches": len(results), "stocks": results})


FUNCTIONS = {
    "get_stock_price": get_stock_price,
    "get_historical_data": get_historical_data,
    "calculate_indicators": calculate_indicators,
    "screen_stocks": screen_stocks,
}

SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price and basic info (PE ratio, market cap, sector, 52-week range) for a stock ticker.",
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
            "description": "Fetch historical OHLCV price data for a stock. Returns up to 20 most recent data points.",
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
            "description": "Calculate technical indicators: RSI (14), SMA (20 & 50), MACD, and trend direction.",
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
            "name": "screen_stocks",
            "description": "Screen 40 major US stocks to find buy/sell candidates. Filter by sector, technical signal, and market cap. Use when the user asks 'what should I buy?' or wants stock recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sector": {
                        "type": "string",
                        "description": "Filter by sector (e.g. Technology, Healthcare, Energy, Financial, Consumer). Leave empty for all sectors.",
                        "default": "",
                    },
                    "signal": {
                        "type": "string",
                        "enum": ["any", "oversold", "overbought", "bullish", "bearish"],
                        "description": "Technical signal filter: oversold (RSI<30), overbought (RSI>70), bullish (SMA20>SMA50), bearish (SMA20<SMA50), or any",
                        "default": "any",
                    },
                    "min_market_cap": {
                        "type": "string",
                        "enum": ["any", "large", "mid"],
                        "description": "Market cap filter: large (>100B), mid (10B-100B), or any",
                        "default": "any",
                    },
                },
            },
        },
    },
]
