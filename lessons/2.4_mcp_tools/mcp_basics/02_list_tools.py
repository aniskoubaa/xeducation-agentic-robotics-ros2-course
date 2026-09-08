"""The other half of MCP: a client that connects, DISCOVERS, then calls.

Nothing here is imported from the server - it is a separate process, and the
only thing between them is the protocol.  The host asks "what tools do you
have?" at connect time, which is why nothing is ever hard-coded.

    python3 02_list_tools.py
"""

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# how to START the server: a command, exactly like launching any other program
SERVER = StdioServerParameters(command=sys.executable, args=['01_hello_server.py'])


async def main():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()               # the handshake

            # ---- DISCOVER: the tool list comes from the server ------------
            tools = (await session.list_tools()).tools
            print(f'discovered {len(tools)} tools\n')
            for t in tools:
                args = ', '.join(f'{k}: {v["type"]}'
                                 for k, v in t.inputSchema['properties'].items())
                print(f'  {t.name:20} ({args})')     # <- from the type hints
                print(f'  {"":20}  {t.description}')  # <- from the docstring

            # ---- CALL: name plus arguments, and the server runs it --------
            print()
            for name, args in [('get_gripper_opening', {}),
                               ('close_gripper', {}),
                               ('get_gripper_opening', {})]:
                out = await session.call_tool(name, args)
                print(f'  call {name:22} -> {out.content[0].text}')


asyncio.run(main())
