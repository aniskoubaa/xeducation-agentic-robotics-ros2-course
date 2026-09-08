"""The same tool with Gemini - the SDK drives the round trip for you.

Hand it the plain Python function and it does by itself what we wrote out by
hand in 01: ask for the tool, run it, feed the result back, answer.

    python3 02_gemini_tools.py
"""

import logging
import os

from google import genai
from google.genai import types

MODEL = 'gemini-2.5-flash'
# the SDK prints a long advisory note on every call - keep the output clean
logging.getLogger('google_genai').setLevel(logging.ERROR)

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])


def get_gripper_opening() -> dict:
    """Read how wide the robot gripper is open, in millimetres."""
    return {'opening_mm': 42.0}


# the docstring and the type hints ARE the tool description here
reply = client.models.generate_content(
    model=MODEL,
    config=types.GenerateContentConfig(tools=[get_gripper_opening]),
    contents='How wide is the gripper open right now?',
)

# the SDK already called the function before we got here
for step in reply.automatic_function_calling_history:
    for part in step.parts:
        if part.function_call:
            print('the model called   :', part.function_call.name)
        if part.function_response:
            print('the function gave  :', part.function_response.response)

print('the model answers  :', reply.text)
