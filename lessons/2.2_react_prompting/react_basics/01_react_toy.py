"""ReAct: Thought -> Action -> Observation, then repeat.

The agent in 2.1 acted, but it never said WHY.  Here we change one thing: every
tool takes an extra `thought` argument, so the model must write its reasoning
before it is allowed to act.  That trace is how you understand - and debug - an
agent.

    python3 01_react_toy.py
"""

import json
import os

from openai import OpenAI

MODEL = 'gpt-4o-mini'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# ---- the robot: two skills ------------------------------------------------
opening_mm = 42.0


def close_gripper():
    """Close the robot gripper."""
    global opening_mm
    opening_mm = 8.0
    return {'closed': True}


def get_gripper_opening():
    """Read how wide the robot gripper is open, in millimetres."""
    return {'opening_mm': opening_mm}


SKILLS = {f.__name__: f for f in (close_gripper, get_gripper_opening)}


def schema(fn):
    """Every tool gains one argument: the reason for calling it."""
    return {'type': 'function', 'function': {
        'name': fn.__name__, 'description': fn.__doc__,
        'parameters': {'type': 'object', 'properties': {
            'thought': {'type': 'string',
                        'description': 'Why, in one SHORT sentence.'}},
            'required': ['thought']}}}


TOOLS = [schema(f) for f in SKILLS.values()]

SYSTEM_PROMPT = ('You are a robot gripper controller. Use the tools to do the '
                 'work - never just describe what you would do. Take ONE action '
                 'at a time, and put the reason in the thought argument.')

messages = [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': 'Close the gripper, then tell me how wide it is.'}]

# ---- the ReAct loop -------------------------------------------------------
for step in range(1, 6):
    reply = client.chat.completions.create(model=MODEL, messages=messages,
                                           tools=TOOLS, parallel_tool_calls=False)
    msg = reply.choices[0].message
    messages.append(msg)

    if not msg.tool_calls:
        print('Final      :', msg.content)
        break

    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments)
    print('Thought    :', args['thought'])         # REASON - written first
    print('Action     :', call.function.name + '()')
    result = SKILLS[call.function.name]()          # ACT   - only now
    print('Observation:', result)                  # OBSERVE
    messages.append({'role': 'tool', 'tool_call_id': call.id,
                     'content': json.dumps(result)})
