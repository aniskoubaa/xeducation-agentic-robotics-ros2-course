"""A whole MCP server, in a few lines.

FastMCP turns a plain function into a tool: the docstring becomes the tool's
description, and the type hints become its input schema.  Nothing in this file
knows about a language model - a server only serves tools.

You never run this one by hand: a client starts it.  See 02_list_tools.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP('toy-gripper', log_level='ERROR')   # one server, many tools

STATE = {'opening_mm': 42.0}


@mcp.tool()
def close_gripper() -> str:
    """Close the gripper on whatever is in front of it."""
    STATE['opening_mm'] = 8.0
    return 'gripper closed'


@mcp.tool()
def open_gripper(width_mm: float) -> str:
    """Open the gripper to a given width, in millimetres."""
    STATE['opening_mm'] = width_mm
    return f'gripper opened to {width_mm} mm'


@mcp.tool()
def get_gripper_opening() -> str:
    """Report how wide the gripper is open, in millimetres."""
    return f"the gripper is {STATE['opening_mm']} mm open"


mcp.run(transport='stdio')               # serve it: JSON-RPC over stdin/stdout
