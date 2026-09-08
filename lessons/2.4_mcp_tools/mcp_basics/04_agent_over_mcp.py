"""The agent from lesson 2.2 again - but it imports no skill at all.

The loop is unchanged.  Two things are different, and they are the whole
lesson: the tool list is DISCOVERED from the server at connect time, and
every Action is session.call_tool() instead of a Python call.

    python3 04_agent_over_mcp.py
"""

import asyncio
import json
import os
import sys
from datetime import timedelta

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

MODEL = 'gpt-4o-mini'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
# MCP starts the server with a CLEAN environment - pass ours, or it has no ROS 2
SERVER = StdioServerParameters(command=sys.executable, args=['03_raisebot_server.py'],
                               env=dict(os.environ))

SYSTEM_PROMPT = (
    'You are the greenhouse inspection robot. '
    'To inspect a plant: navigate there, raise the arm to above_plant, then '
    'inspect_plant. Use the tools - never just describe what you would do. '
    'Take ONE action at a time. Report the result in ONE short sentence.')

GOAL = 'Is the tomato in row 1 healthy?'
TIMEOUT = timedelta(seconds=180)      # driving across the greenhouse takes time


def as_openai(tool):
    """One MCP tool -> one OpenAI tool schema. The protocol already carries both."""
    return {'type': 'function',
            'function': {'name': tool.name, 'description': tool.description,
                         'parameters': tool.inputSchema}}


async def main():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            print('discovered :', ', '.join(t.name for t in tools))
            print('Goal       :', GOAL)

            messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
                        {'role': 'user', 'content': GOAL}]

            # ---- the same loop as 2.2, over the protocol -------------------
            for step in range(1, 8):
                reply = client.chat.completions.create(
                    model=MODEL, messages=messages, tools=[as_openai(t) for t in tools],
                    parallel_tool_calls=False)
                msg = reply.choices[0].message
                messages.append(msg)

                if not msg.tool_calls:
                    print('Final      :', msg.content)
                    break

                c = msg.tool_calls[0]
                args = json.loads(c.function.arguments)
                print('Action     :', c.function.name, args)
                out = await session.call_tool(c.function.name, args,
                                              read_timeout_seconds=TIMEOUT)
                text = out.content[0].text
                print('Observation:', ' '.join(text.split())[:58])
                messages.append({'role': 'tool', 'tool_call_id': c.id, 'content': text})


asyncio.run(main())
