# Lesson 2.1 — an agent = an LLM + tools + a loop

Three scripts, one per piece of that definition. They continue straight on from
lesson 2.0, where the same model could not answer this question at all.

| file | piece | what it shows |
|---|---|---|
| `01_openai_tools.py` | + tools | the round trip written out by hand: ask → run → tell |
| `02_gemini_tools.py` | + tools | the same tool, with the SDK driving the round trip |
| `03_agent_loop.py` | + loop | two skills and a `for` loop: act, observe, decide again |

```bash
pip install openai google-genai
export OPENAI_API_KEY=...        # never put a key in the source
export GEMINI_API_KEY=...
python3 01_openai_tools.py
```

The model never runs a tool. It *asks* for one; your Python runs it and hands
the result back. Do that once and you have tool calling; do it in a loop, one
action at a time, and you have an agent.
