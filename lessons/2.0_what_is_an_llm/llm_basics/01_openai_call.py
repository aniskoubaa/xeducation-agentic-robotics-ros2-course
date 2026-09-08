"""Rung 1 of the ladder: a bare LLM call.  Text in -> text out.

    export OPENAI_API_KEY=...        # never write the key in the file
    python3 01_openai_call.py

That is the whole thing: one function call, one string back.  Everything else
in this module - tools, memory, the agent loop - is built on top of this.
"""

import os

from openai import OpenAI

MODEL = 'gpt-4o-mini'

# the key is read from the environment, so it never appears in the source
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

reply = client.chat.completions.create(
    model=MODEL,
    messages=[
        # the SYSTEM message is the role and the rules - rung 2 of the ladder
        {'role': 'system',
         'content': 'You are a robotics teaching assistant. '
                    'Answer in one short sentence.'},
        # the USER message is the question we are asking right now
        {'role': 'user',
         'content': 'What is a large language model?'},
    ],
)

# the answer is plain text, nothing more
print(reply.choices[0].message.content)
