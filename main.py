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
