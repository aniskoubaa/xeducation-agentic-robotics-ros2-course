"""An agent = an LLM + tools + a LOOP.

01 did one round trip by hand: ask, run, tell.  An agent does not stop there.
It looks at the result and decides again, one action at a time, until it has
an answer.  That while-loop is the only thing 01 was missing.

    python3 03_agent_loop.py
"""

import json
import os

from openai import OpenAI

MODEL = 'gpt-4o-mini'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# ---- the robot: two skills, and the state they share ----------------------
opening_mm = 42.0


def close_gripper():
    """Close the robot gripper."""
    global opening_mm
    opening_mm = 8.0
    return {'closed': True}


def get_gripper_opening():
    """Read how wide the robot gripper is open, in millimetres."""
    return {'opening_mm': opening_mm}


SKILLS = {'close_gripper': close_gripper,
          'get_gripper_opening': get_gripper_opening}

# each docstring becomes the description the model reads
TOOLS = [{'type': 'function',
          'function': {'name': name, 'description': fn.__doc__,
                       'parameters': {'type': 'object', 'properties': {}}}}
         for name, fn in SKILLS.items()]

messages = [
    # "one action at a time" is what makes the loop visible - and honest
    {'role': 'system', 'content': 'You are a robot. Take one action at a time.'},
    {'role': 'user', 'content': 'Close the gripper, then tell me how wide it is.'},
]

# ---- the loop: act, observe, decide again ---------------------------------
for step in range(1, 6):
    reply = client.chat.completions.create(model=MODEL, messages=messages,
                                          tools=TOOLS, parallel_tool_calls=False)
    msg = reply.choices[0].message
    messages.append(msg)                     # the history IS the agent's memory

    if not msg.tool_calls:                   # nothing left to call - it is done
        print(f'{step}. answer :', msg.content)
        break

    for call in msg.tool_calls:              # ACT, then OBSERVE
        result = SKILLS[call.function.name]()
        print(f'{step}. call   : {call.function.name}() -> {result}')
        messages.append({'role': 'tool', 'tool_call_id': call.id,
                         'content': json.dumps(result)})
