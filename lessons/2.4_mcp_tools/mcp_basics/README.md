# Lesson 2.4 — MCP: one standard way to share tools

Two halves. First a toy server and a client, so the protocol is visible on its
own. Then the same two files wrapping the real robot skills.

| file | role | what it shows |
|---|---|---|
| `01_hello_server.py` | server | three tools, from three plain functions |
| `02_list_tools.py` | client | connect → **discover** → call, over stdio |
| `03_raisebot_server.py` | server | the identical shape, each tool a ROS 2 `Trigger` service |
| `04_agent_over_mcp.py` | client | the lesson 2.2 agent, with its tools discovered instead of imported |

The client and the server are **separate processes**. Nothing is imported
across the line — `02` never sees `01`'s functions, it asks for them.

```bash
pip install "mcp<2"              # 1.x is the FastMCP API used on slide 4

python3 02_list_tools.py         # no robot needed; it starts 01 for you

export OPENAI_API_KEY=...        # never put a key in the source
raise-sim                        # 04 needs the sim + the skill servers
ros2 run raise2026_tools navigation_server &
ros2 run raise2026_tools move_to_pose_server &
ros2 run raise2026_tools inspector_server &
python3 04_agent_over_mcp.py     # it starts 03 for you
```

**What `@mcp.tool()` does:** the docstring becomes the tool's description and
the type hints become its input schema. That is the same pair of things you
hand-wrote as a JSON schema in lesson 2.3 — FastMCP derives it from the
function itself.

**What changes in the agent:** nothing in the loop. `04` is `02_react_robot.py`
from lesson 2.2 with two lines swapped — the tool list comes from
`session.list_tools()`, and each action is `session.call_tool()`. The server
knows nothing about prompts, and the agent knows nothing about ROS 2.

> **SDK note.** MCP 2.x renamed `FastMCP` to `MCPServer`. These files use the
> 1.x name, matching the deck; pin `mcp<2` and they run as written.
