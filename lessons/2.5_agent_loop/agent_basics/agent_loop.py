"""The parts that never change: how to declare a tool, how to ask, how to loop.

Nothing in this file knows about any robot.  It is the same for every robot you
will ever build, which is why it lives on its own.  The two robot-specific
pieces - the tool LIST and the MAPPING - are handed in by the caller.

Both 01_dry_run.py and 02_raise_bot.py import this file, unchanged.
"""

import json
import os

from openai import OpenAI

MODEL = 'gpt-4o-mini'
MAX_STEPS = 8                      # guardrail: the loop is bounded, always
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


def tool(name, description, **args):
    """Piece 1 - declare: a name, one sentence, and the arguments it takes."""
    props = {k: {'type': 'string', 'description': v} for k, v in args.items()}
    return {'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': props,
                       'required': list(props)}}}


def ask_llm(messages, tools):
    """One turn of the model: either a tool call, or a final answer."""
    return client.chat.completions.create(
        model=MODEL, messages=messages, tools=tools,
        parallel_tool_calls=False).choices[0].message


def run(system_prompt, goal, tools, tool_to_service, call_service):
    """Piece 4 - the loop: the model picks, your code runs, the result feeds back."""
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': goal}]
    print('Goal       :', goal)

    for step in range(1, MAX_STEPS + 1):        # bounded - never runs forever
        msg = ask_llm(messages, tools)
        messages.append(msg)

        if not msg.tool_calls:                  # the model decided it is done
            print('Final      :', msg.content)
            return

        c = msg.tool_calls[0]
        args = json.loads(c.function.arguments)
        print(f'Action {step}   :', c.function.name, args)
        service = tool_to_service(c.function.name, args)      # 2 - map
        print('Service    :', service)
        result = call_service(service)                        # 3 - call
        print('Observation:', ' '.join(result.split())[:58])
        # the observation goes back into the messages - that IS the memory
        messages.append({'role': 'tool', 'tool_call_id': c.id, 'content': result})

    print('Stopped    : hit MAX_STEPS =', MAX_STEPS)
