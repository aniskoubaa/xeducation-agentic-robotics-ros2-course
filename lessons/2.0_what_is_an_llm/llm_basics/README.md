# Lesson 2.0 — a first LLM call

Three tiny scripts: what an LLM call *is*, and what it cannot do on its own.

| file | what it shows |
|---|---|
| `01_openai_call.py` | one call — text in, text out |
| `02_gemini_call.py` | the same call, a different provider |
| `03_no_tools.py` | the three gaps: it cannot act, know, or remember |

```bash
pip install openai google-genai
export OPENAI_API_KEY=...        # never put a key in the source
export GEMINI_API_KEY=...
python3 01_openai_call.py
```

An LLM call is one function call: you send text, you get text back. Filling the
three gaps — with tools, a prompt and a loop — is what turns it into an agent.
That is lesson 2.1, in `lessons/2.1_agentic_ai/agent_basics/`.
