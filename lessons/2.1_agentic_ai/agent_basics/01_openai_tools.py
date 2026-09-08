"""A tool: the same question as lesson 2.0, but now the model can ask.

The model never runs the tool itself.  It asks us to run it, we run the plain
Python function, we hand the result back, and only then can it answer.

    ask  ->  the model requests get_gripper_opening()
    run  ->  WE call the function
    tell ->  the model answers with the real number

    python3 01_openai_tools.py
"""

import json
import os

from openai import OpenAI

MODEL = 'gpt-4o-mini'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


def get_gripper_opening():
    """The 'robot'.  In lesson 1.12 this was a real ROS 2 service call."""
    return {'opening_mm': 42.0}


# the model never sees the code - only this description of it
TOOLS = [{
    'type': 'function',
    'function': {
        'name': 'get_gripper_opening',
        'description': 'Read how wide the robot gripper is open, in mm.',
        'parameters': {'type': 'object', 'properties': {}},
    },
}]

messages = [{'role': 'user', 'content': 'How wide is the gripper open?'}]

# 1) ASK - the model cannot know, so it requests the tool instead of guessing
first = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
call = first.choices[0].message.tool_calls[0]
print('1. the model asked for :', call.function.name)

# 2) RUN - this line is ours, not the model's.  Here is where a robot moves.
result = get_gripper_opening()
print('2. we ran it, it gave  :', result)

# 3) TELL - put the result in the conversation and ask again
messages.append(first.choices[0].message)
messages.append({'role': 'tool', 'tool_call_id': call.id,
                 'content': json.dumps(result)})
second = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
print('3. the model answers   :', second.choices[0].message.content)
