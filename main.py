"""
Terminal REPL for the Multi-Agent Stock Trading System.

Run with:  python main.py

A lightweight text interface — no browser needed. Useful for quick testing.
Specialist agents are stateless in this mode (no Chainlit session).
"""

import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

import opik
opik.configure(use_local=False)

from orchestrator import create_client, create_messages, run_orchestrator


def print_agent_call(fn_name, fn_args, result):
    print(f"\n  📋 Orchestrator → {fn_name}")


async def async_main():
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

    client = create_client()
    messages = create_messages()

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        response = await run_orchestrator(client, messages, on_agent_call=print_agent_call)
        print(f"\n🤖 Orchestrator: {response}\n")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
