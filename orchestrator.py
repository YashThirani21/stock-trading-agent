"""
Shared orchestrator loop — the "boss" agent that coordinates specialists.

Used by both app.py (Chainlit web UI) and main.py (terminal REPL).
Each frontend passes an optional callback to hook into agent calls
(e.g. terminal print statements).
"""

import json
from openai import AsyncOpenAI
from opik import track
from opik.integrations.openai import track_openai

from agent_loop import _stream_completion
from agents import AGENT_SCHEMAS, AGENT_FUNCTIONS
from config import MODEL, MAX_TURNS_ORCHESTRATOR, ORCHESTRATOR_PROMPT


def create_client():
    return track_openai(AsyncOpenAI())


def create_messages():
    return [{"role": "system", "content": ORCHESTRATOR_PROMPT}]


@track(name="orchestrator", capture_input=True, capture_output=True)
async def run_orchestrator(client, messages, on_agent_call=None, on_token=None):
    """
    Run one orchestrator turn — may involve multiple specialist agent calls.

    on_agent_call(fn_name, fn_args, result) is called after each specialist
    returns, so frontends can log trades, print status, etc.
    """
    for turn in range(MAX_TURNS_ORCHESTRATOR):
        kwargs = {
            "model": MODEL,
            "messages": messages,
            "tools": AGENT_SCHEMAS,
            "tool_choice": "auto",
        }
        content, tc_list, msg_dict = await _stream_completion(client, kwargs, on_token=on_token)
        messages.append(msg_dict)

        if not tc_list:
            return content or ""

        for tc in tc_list:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])

            if fn_name in AGENT_FUNCTIONS:
                result = await AGENT_FUNCTIONS[fn_name](**fn_args)
            else:
                result = f"Unknown agent: {fn_name}"

            if on_agent_call:
                on_agent_call(fn_name, fn_args, result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    return "Orchestrator reached max turns."
