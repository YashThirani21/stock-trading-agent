"""
Shared orchestrator loop — the "boss" agent that coordinates specialists.

Used by both app.py (Chainlit web UI) and main.py (terminal REPL).
Each frontend passes an optional callback to hook into agent calls
(e.g. trade logging for the sidebar, or terminal print statements).
"""

import json
from openai import AsyncOpenAI
from opik import track
from opik.integrations.openai import track_openai

from agents import AGENT_SCHEMAS, AGENT_FUNCTIONS
from config import MODEL, MAX_TURNS_ORCHESTRATOR, ORCHESTRATOR_PROMPT


def create_client():
    return track_openai(AsyncOpenAI())


def create_messages():
    return [{"role": "system", "content": ORCHESTRATOR_PROMPT}]


@track(name="orchestrator", capture_input=True, capture_output=True)
async def run_orchestrator(client, messages, on_agent_call=None):
    """
    Run one orchestrator turn — may involve multiple specialist agent calls.

    on_agent_call(fn_name, fn_args, result) is called after each specialist
    returns, so frontends can log trades, print status, etc.
    """
    for turn in range(MAX_TURNS_ORCHESTRATOR):
        response = await client.chat.completions.create(
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

            if fn_name in AGENT_FUNCTIONS:
                result = await AGENT_FUNCTIONS[fn_name](**fn_args)
            else:
                result = f"Unknown agent: {fn_name}"

            if on_agent_call:
                on_agent_call(fn_name, fn_args, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Orchestrator reached max turns."
