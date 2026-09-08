"""The same call, a different provider.  Same idea: text in -> text out.

    export GEMINI_API_KEY=...
    python3 02_gemini_call.py

The words change - `contents` instead of `messages`, `system_instruction`
instead of a system message - but the shape is identical.  Every provider
gives you the same thing: you send text, you get text back.
"""

import logging
import os

from google import genai
from google.genai import types

MODEL = 'gemini-2.5-flash'

# the SDK prints a long advisory note on every call - keep the output clean
logging.getLogger('google_genai').setLevel(logging.ERROR)

client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])

reply = client.models.generate_content(
    model=MODEL,
    # the role and the rules - the same job the system message did above
    config=types.GenerateContentConfig(
        system_instruction='You are a robotics teaching assistant. '
                           'Answer in one short sentence.'),
    # the question
    contents='What is a large language model?',
)

print(reply.text)
