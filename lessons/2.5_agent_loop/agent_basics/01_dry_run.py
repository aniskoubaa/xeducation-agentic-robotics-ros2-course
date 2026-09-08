"""The two robot-specific pieces - on a robot that is not plugged in.

DECLARE the tools, MAP each name to a service, then hand both to the loop.
call_service here never touches ROS 2: it just says "ok".  That is enough to
build and debug the prompt, the tool list and the mapping on your laptop,
before you ever start the simulator.

    python3 01_dry_run.py
"""

from agent_loop import run, tool

# ---- 1. DECLARE: the only things the model is allowed to ask for -----------
TOOLS = [
    tool('navigate', 'Drive the robot to a named greenhouse waypoint.',
         destination='home, tomato_row_1, tomato_row_2, olive_grove'),
    tool('move_arm', 'Move the arm to a named pose.',
         pose='home, above_plant, side_view, stow'),
    tool('operate_gripper', 'Open or close the gripper.',
         action='open, close'),
]


# ---- 2. MAP: the model's (name, args) -> one ROS 2 service name ------------
def tool_to_service(name, args):
    if name == 'navigate':        return f"/nav_to_{args['destination']}"
    if name == 'move_arm':        return f"/move_to_{args['pose']}"
    if name == 'operate_gripper': return f"/{args['action']}_gripper"
    raise ValueError(f'unknown tool {name}')


# ---- 3. CALL: nothing is running today, so just say it worked --------------
def call_service(service):
    return 'ok (dry run)'


SYSTEM_PROMPT = (
    'You are the greenhouse robot. To pick a plant: navigate there, move the '
    'arm to above_plant, then close the gripper. Use the tools - never just '
    'describe what you would do. Take ONE action at a time. '
    'Report the result in ONE short sentence.')

GOAL = 'Pick the ripe tomato in row 2.'

run(SYSTEM_PROMPT, GOAL, TOOLS, tool_to_service, call_service)     # 4. LOOP
