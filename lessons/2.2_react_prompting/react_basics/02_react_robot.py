"""The same ReAct loop - but every Action is a real ROS 2 service call.

Nothing changes except the SKILLS table: the toy functions are replaced by
Trigger services on the running robot.  Watch Gazebo while the trace prints.

    python3 02_react_robot.py
"""

import json
import os

import rclpy
from openai import OpenAI
from rclpy.node import Node
from std_srvs.srv import Trigger

MODEL = 'gpt-4o-mini'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# ---- the tools: one Trigger service each ----------------------------------
WAYPOINTS = ['home', 'tomato_row_1', 'tomato_row_2', 'olive_grove']
ARM_POSES = ['home', 'above_plant', 'side_view', 'stow']


def call(node, service):
    """Call one std_srvs/Trigger service and return its reply message."""
    cli = node.create_client(Trigger, service)
    cli.wait_for_service(timeout_sec=10.0)
    fut = cli.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(node, fut, timeout_sec=120.0)
    res = fut.result()
    return res.message if res else 'no reply'


def navigate(node, a):
    """Drive the robot base to a named greenhouse waypoint."""
    return call(node, f"/nav_to_{a['destination']}")


def move_arm(node, a):
    """Move the arm to a named pose."""
    return call(node, f"/move_to_{a['pose']}")


def inspect_plant(node, a):
    """Look at the plant with the camera and judge its health."""
    return call(node, '/inspect_plant')


SKILLS = {'navigate': navigate, 'move_arm': move_arm, 'inspect_plant': inspect_plant}
ENUMS = {'navigate': ('destination', WAYPOINTS), 'move_arm': ('pose', ARM_POSES)}


def schema(fn):
    """Same shape as the toy version: a thought, plus this tool's own argument."""
    props = {'thought': {'type': 'string', 'description': 'Why, in one SHORT sentence.'}}
    if fn.__name__ in ENUMS:
        arg, values = ENUMS[fn.__name__]
        props[arg] = {'type': 'string', 'enum': values}
    return {'type': 'function', 'function': {
        'name': fn.__name__, 'description': fn.__doc__,
        'parameters': {'type': 'object', 'properties': props,
                       'required': list(props)}}}


TOOLS = [schema(f) for f in SKILLS.values()]

# the system prompt grounds the agent: who it is, its world, and the rules
SYSTEM_PROMPT = (
    'You are the greenhouse inspection robot. '
    'Waypoints: home, tomato_row_1, tomato_row_2, olive_grove. '
    'To inspect a plant: navigate there, raise the arm to above_plant, then '
    'inspect_plant. Use the tools - never just describe what you would do. '
    'Take ONE action at a time. Report the result in ONE short sentence.')

GOAL = 'Is the tomato in row 1 healthy?'

rclpy.init()
node = Node('react_demo')
messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': GOAL}]
print('Goal       :', GOAL)

# ---- the ReAct loop - identical to 01, only the SKILLS changed -------------
for step in range(1, 8):
    reply = client.chat.completions.create(model=MODEL, messages=messages,
                                           tools=TOOLS, parallel_tool_calls=False)
    msg = reply.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        print('Final      :', msg.content)
        break

    c = msg.tool_calls[0]
    args = json.loads(c.function.arguments)
    print('Thought    :', args['thought'])
    print('Action     :', c.function.name, {k: v for k, v in args.items() if k != 'thought'})
    result = SKILLS[c.function.name](node, args)
    print('Observation:', ' '.join(result.split())[:58])
    messages.append({'role': 'tool', 'tool_call_id': c.id, 'content': result})

node.destroy_node()
rclpy.shutdown()
