"""What the model CANNOT do on its own: act, know, remember.

Ask it something only the robot knows.  It has no sensors, no hands and no
memory of the last call - so all it can do is talk about the question.

    python3 03_no_tools.py
"""

import os

from openai import OpenAI

MODEL = 'gpt-4o-mini'
RULES = 'You are a robot assistant. Answer in one short sentence.'

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


def ask(question):
    """One call = one question.  Nothing is carried over between calls."""
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[{'role': 'system', 'content': RULES},
                  {'role': 'user', 'content': question}],
    )
    return reply.choices[0].message.content


# 1) it cannot KNOW - there is no sensor behind this call
q1 = 'How wide is the gripper open right now, in millimetres?'
print('Q1:', q1)
print('A1:', ask(q1))

# 2) it cannot REMEMBER - this is a brand new call, it starts from zero
q2 = 'Do you remember what I asked you a moment ago?'
print('Q2:', q2)
print('A2:', ask(q2))
