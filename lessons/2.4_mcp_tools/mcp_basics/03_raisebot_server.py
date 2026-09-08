"""The RAISE-Bot MCP server: our robot skills, behind the protocol.

Same shape as 01 - only the tool bodies changed.  Each one is the ROS 2
Trigger service you already built in Module 1, so MCP does not replace the
skills, it wraps them so that any host can reach them.

Started by 04_agent_over_mcp.py.  The sim must be running.
"""

import rclpy
from mcp.server.fastmcp import FastMCP
from rclpy.node import Node
from std_srvs.srv import Trigger

mcp = FastMCP('raise-bot', log_level='ERROR')

rclpy.init()
node = Node('raisebot_mcp_server')


def call(service):
    """Call one std_srvs/Trigger service and return its reply message."""
    cli = node.create_client(Trigger, service)
    cli.wait_for_service(timeout_sec=10.0)
    fut = cli.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=120.0)
    res = fut.result()
    return res.message if res else 'no reply'


@mcp.tool()
def navigate(destination: str) -> str:
    """Drive the robot to a waypoint: home, tomato_row_1, tomato_row_2, olive_grove."""
    return call(f'/nav_to_{destination}')


@mcp.tool()
def move_arm(pose: str) -> str:
    """Move the arm to a pose: home, above_plant, side_view, stow."""
    return call(f'/move_to_{pose}')


@mcp.tool()
def inspect_plant() -> str:
    """Look at the plant with the camera and judge its health."""
    return call('/inspect_plant')


mcp.run(transport='stdio')
