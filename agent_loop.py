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
from config import MODEL_SPECIALIST, MAX_TURNS_SPECIALIST


def _has_chainlit_context() -> bool:
    try:
        cl.context.session
        return True
    except Exception:
        return False


async def _stream_completion(client, kwargs, on_token=None):
    stream = await client.chat.completions.create(
        **kwargs, stream=True, stream_options={"include_usage": True}
    )

    content = ""
    tool_calls = {}

    async for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta

        if delta.content:
            content += delta.content
            if on_token:
                await on_token(delta.content)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tool_calls:
                    tool_calls[idx] = {
                        "id": tc_delta.id,
                        "type": "function",
                        "function": {"name": tc_delta.function.name, "arguments": ""},
                    }
                if tc_delta.function and tc_delta.function.arguments:
                    tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

    msg_dict = {"role": "assistant", "content": content or None}
    if tool_calls:
        msg_dict["tool_calls"] = list(tool_calls.values())

    return content or None, list(tool_calls.values()) if tool_calls else None, msg_dict


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
    parent_step=None,
    response_schema: Optional[dict] = None,
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
        kwargs = {"model": MODEL_SPECIALIST, "messages": messages}
        if tool_schemas:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "auto"

        on_token = parent_step.stream_token if parent_step else None
        content, tc_list, msg_dict = await _stream_completion(client, kwargs, on_token=on_token)
        messages.append(msg_dict)

        if not tc_list:
            if response_schema and content:
                format_msgs = [
                    {"role": "system", "content": "Convert the following analysis into the required JSON format. Preserve all information accurately."},
                    {"role": "user", "content": content},
                ]
                format_response = await client.chat.completions.create(
                    model=MODEL_SPECIALIST,
                    messages=format_msgs,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "response",
                            "strict": True,
                            "schema": response_schema,
                        },
                    },
                )
                return format_response.choices[0].message.content or content
            return content or ""

        for tc in tc_list:
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])

            if _has_chainlit_context():
                async with cl.Step(name=fn_name, type="tool") as step:
                    step.input = json.dumps(fn_args, default=str)
                    await step.send()
                    result = await asyncio.to_thread(
                        execute_tool, fn_name, fn_args, tool_functions
                    )
                    step.output = result
            else:
                result = await asyncio.to_thread(
                    execute_tool, fn_name, fn_args, tool_functions
                )

            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

    return "Agent reached max turns without a final response."
