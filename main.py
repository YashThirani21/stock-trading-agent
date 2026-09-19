"""
Multi-Agent Stock Trading System — entry point.

The orchestrator is the "boss" agent. It doesn't call market APIs or place trades
directly. Instead, its tools are the 4 specialist agents:

    User
     │
     ▼
  Orchestrator (GPT-4o-mini)
     │
     ├── call_market_analyst(query)  → runs a full agent loop, returns analysis
     ├── call_news_analyst(query)    → runs a full agent loop, returns sentiment
     ├── call_risk_manager(query)    → runs a full agent loop, returns risk assessment
     └── call_trader(query)          → runs a full agent loop, returns execution result

Each specialist has its own system prompt, its own tools, and its own isolated context.
The orchestrator coordinates them by passing context between calls.

Observability: all LLM calls, tool calls, and agent calls are traced via Opik.
View traces at: https://www.comet.com/opik
"""

import json
import os

from dotenv import load_dotenv
load_dotenv()

import opik
opik.configure(use_local=False)

from openai import OpenAI
from opik import track
from opik.integrations.openai import track_openai
from agents import AGENT_SCHEMAS, AGENT_FUNCTIONS

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


@track(name="orchestrator", capture_input=True, capture_output=True)
def run_orchestrator_turn(client, messages):
    """
    Run one orchestrator turn — may involve multiple specialist agent calls.
    This is the top-level trace. Everything nested inside (specialist agents,
    their LLM calls, their tool calls) appears as child spans.
    """
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

            print(f"\n  📋 Orchestrator → {fn_name}")

            if fn_name in AGENT_FUNCTIONS:
                result = AGENT_FUNCTIONS[fn_name](**fn_args)
            else:
                result = f"Unknown agent: {fn_name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Orchestrator reached max turns."


def main():
    print("=" * 60)
    print("  Multi-Agent Stock Trading System")
    print("  Agents: Market Analyst | News Analyst | Risk Manager | Trader")
    print("  Observability: Opik (https://www.comet.com/opik)")
    print("  Type 'quit' to exit")
    print("=" * 60)

    missing = []
    for key in ["OPENAI_API_KEY", "ALPACA_API_KEY", "ALPACA_SECRET_KEY", "FINNHUB_API_KEY"]:
        if not os.environ.get(key):
            missing.append(key)
    if missing:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing)}")
        print("  Add them to your .env file")
        return

    client = track_openai(OpenAI())
    messages = [{"role": "system", "content": ORCHESTRATOR_PROMPT}]

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        response = run_orchestrator_turn(client, messages)
        print(f"\n🤖 Orchestrator: {response}\n")


if __name__ == "__main__":
    main()
