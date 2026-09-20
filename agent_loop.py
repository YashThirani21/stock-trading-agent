"""
Generic ReAct agent loop — reused by every specialist agent and the orchestrator.

Now instrumented with:
- Opik for observability (traces to dashboard)
- Chainlit for UI (tool calls appear as expandable steps in the chat)

Supports both stateless (fresh messages each call) and stateful (persistent
messages across calls) modes via the optional `messages` parameter.
"""

import asyncio
import json
from typing import Optional
from openai import AsyncOpenAI
from opik import track
from opik.integrations.openai import track_openai
import chainlit as cl
from config import MODEL, MAX_TURNS_SPECIALIST


@track(capture_input=True, capture_output=True)
def execute_tool(fn_name: str, fn_args: dict, tool_functions: dict) -> str:
    if fn_name in tool_functions:
        result = tool_functions[fn_name](**fn_args)
    else:
        result = json.dumps({"error": f"Unknown tool: {fn_name}"})
    return result if isinstance(result, str) else json.dumps(result)


@track(capture_input=True, capture_output=True)
async def run_agent(
    system_prompt: str,
    user_message: str,
    tool_schemas: list,
    tool_functions: dict,
    messages: Optional[list] = None,
) -> str:
    """
    Run a complete agent turn: prompt -> tool calls -> final response.

    If `messages` is provided, the agent appends to it (stateful — remembers
    previous calls). If None, creates a fresh conversation (stateless).
    """
    client = track_openai(AsyncOpenAI())

    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
        ]

    messages.append({"role": "user", "content": user_message})

    for turn in range(MAX_TURNS_SPECIALIST):
        kwargs = {"model": MODEL, "messages": messages}
        if tool_schemas:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "auto"

        response = await client.chat.completions.create(**kwargs)
        assistant_message = response.choices[0].message
        messages.append(assistant_message.model_dump())

        if not assistant_message.tool_calls:
            return assistant_message.content or ""

        for tool_call in assistant_message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            async with cl.Step(name=fn_name, type="tool") as step:
                step.input = json.dumps(fn_args, default=str)
                await step.send()
                result = await asyncio.to_thread(
                    execute_tool, fn_name, fn_args, tool_functions
                )
                step.output = result

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Agent reached max turns without a final response."
