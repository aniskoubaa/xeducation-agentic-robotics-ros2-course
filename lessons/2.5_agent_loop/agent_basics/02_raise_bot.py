"""The same two pieces, wired to the real robot.

Compare with 01_dry_run.py: the tool list gained inspect_plant, the mapping
gained one line, and call_service now really calls a ROS 2 service.
agent_loop.py - declare, ask, loop - is not touched at all.

    python3 02_raise_bot.py        # needs the sim + the skill servers
"""

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from agent_loop import run, tool

# ---- 1. DECLARE: the only things the model is allowed to ask for -----------
TOOLS = [
    tool('navigate', 'Drive the robot to a named greenhouse waypoint.',
         destination='home, tomato_row_1, tomato_row_2, olive_grove'),
    tool('move_arm', 'Move the arm to a named pose.',
         pose='home, above_plant, side_view, stow'),
    tool('operate_gripper', 'Open or close the gripper.',
         action='open, close'),
    tool('inspect_plant', 'Look at the plants on one side and judge their health.',
         side='left, right'),
]


# ---- 2. MAP: the model's (name, args) -> one ROS 2 service name ------------
def tool_to_service(name, args):
    if name == 'navigate':        return f"/nav_to_{args['destination']}"
    if name == 'move_arm':        return f"/move_to_{args['pose']}"
    if name == 'operate_gripper': return f"/{args['action']}_gripper"
    if name == 'inspect_plant':   return f"/inspect_{args['side']}"
    raise ValueError(f'unknown tool {name}')


rclpy.init()
node = Node('agent_loop_demo')


# ---- 3. CALL: one std_srvs/Trigger service, and its reply as text ----------
def call_service(service):
    cli = node.create_client(Trigger, service)
    cli.wait_for_service(timeout_sec=10.0)
    fut = cli.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=120.0)
    res = fut.result()
    return res.message if res else 'no reply'


SYSTEM_PROMPT = (
    'You are the greenhouse inspection robot. The crop rows are on your left. '
    'To inspect a plant: navigate there, move the arm to above_plant, then '
    'inspect_plant. Use the tools - never just describe what you would do. '
    'Take ONE action at a time. Report the result in ONE short sentence.')

GOAL = 'Is the tomato in row 1 healthy?'

run(SYSTEM_PROMPT, GOAL, TOOLS, tool_to_service, call_service)     # 4. LOOP

node.destroy_node()
rclpy.shutdown()
