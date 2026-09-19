"""
Stock Trading Agent — the core agent loop.

This is the ReAct (Reason + Act) loop:
1. Send messages + tool schemas to the LLM
2. If the LLM returns tool calls → execute them, append results, loop back
3. If the LLM returns text → that's the final answer, print and stop

         ┌──────────────────────────┐
         │   You type a question    │
         └────────────┬─────────────┘
                      ▼
         ┌──────────────────────────┐
         │   LLM reasons about it   │◄──────────┐
         └────────────┬─────────────┘            │
                      ▼                          │
                ┌───────────┐                    │
                │ Tool call?│── YES ──┐          │
                └─────┬─────┘         ▼          │
                      │      ┌──────────────┐    │
                   NO │      │ Execute tool  │    │
                      │      │ Append result │────┘
                      ▼      └──────────────┘
         ┌──────────────────────────┐
         │   Print final answer     │
         └──────────────────────────┘
"""

import json
import os

from openai import OpenAI
from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
from dotenv import load_dotenv
load_dotenv()
# ── Configuration ────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"
MAX_TURNS = 15  # safety limit to avoid infinite loops

SYSTEM_PROMPT = """You are a stock trading assistant connected to an Alpaca paper trading account with real market data.

Your capabilities:
- Fetch current stock prices and company info
- Retrieve historical price data
- Calculate technical indicators (RSI, SMA, MACD)
- View the Alpaca paper trading portfolio (cash, holdings, P&L)
- Place real paper trades via Alpaca (market orders)
- Check order status after placing trades

Your rules:
1. ALWAYS check technical indicators before recommending or placing a trade
2. ALWAYS check the portfolio before trading to understand current exposure
3. Explain your reasoning — what the indicators suggest and why you're recommending an action
4. Be conservative — these are real orders on a paper account
5. If the user asks you to buy/sell, confirm the analysis supports it before executing
6. Never place a trade without showing the user your analysis first
7. After placing an order, check its status to confirm it was filled
8. Note: market orders only execute during US market hours (9:30 AM - 4:00 PM ET, Mon-Fri)"""


def run_agent(user_message: str, messages: list) -> list:
    """
    Run one full agent turn: send the user message, loop through tool calls,
    return when the LLM produces a final text response.
    """
    client = OpenAI()  # reads OPENAI_API_KEY from environment

    messages.append({"role": "user", "content": user_message})

    turn = 0
    while turn < MAX_TURNS:
        turn += 1

        # ── Step 1: Call the LLM with messages + tools ──
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",  # let the model decide whether to call a tool
        )

        assistant_message = response.choices[0].message

        # Append the assistant's response to conversation history
        messages.append(assistant_message.model_dump())

        # ── Step 2: Check if the LLM wants to call tools ──
        if not assistant_message.tool_calls:
            # No tool calls = final text answer
            print(f"\n🤖 Agent: {assistant_message.content}\n")
            return messages

        # ── Step 3: Execute each tool call ──
        for tool_call in assistant_message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            print(f"  🔧 Calling: {fn_name}({json.dumps(fn_args)})")

            # Look up and execute the tool
            if fn_name in TOOL_FUNCTIONS:
                result = TOOL_FUNCTIONS[fn_name](**fn_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})

            # ── Step 4: Append tool result to messages ──
            # This is how the LLM "sees" the tool output on the next loop
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # Loop back to Step 1 — LLM will reason about the tool results

    print("⚠️  Hit max turns — stopping agent loop.")
    return messages


def main():
    print("=" * 60)
    print("  Stock Trading Agent (Paper Trading)")
    print("  Type 'quit' to exit, 'portfolio' for quick portfolio view")
    print("=" * 60)

    missing = []
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not os.environ.get("ALPACA_API_KEY"):
        missing.append("ALPACA_API_KEY")
    if not os.environ.get("ALPACA_SECRET_KEY"):
        missing.append("ALPACA_SECRET_KEY")
    if missing:
        print(f"\n⚠️  Missing environment variables: {', '.join(missing)}")
        print("  Add them to your .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ALPACA_API_KEY=PK...")
        print("  ALPACA_SECRET_KEY=...")
        return

    # Conversation history persists across turns
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages = run_agent(user_input, messages)


if __name__ == "__main__":
    main()
