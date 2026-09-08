# Lesson 2.5 — the whole agent loop

Four pieces: **declare** the tools, **map** each name to a robot skill, **call**
it, **loop**. Only the first two know anything about your robot.

| file | what it holds | needs |
|---|---|---|
| `agent_loop.py` | declare, ask, loop — the parts that never change | an API key |
| `01_dry_run.py` | the tool list + the mapping, with `call_service` stubbed | an API key |
| `02_raise_bot.py` | the same two pieces, calling real Trigger services | the sim + skill servers |

Both robot files import `agent_loop.py` unchanged — that is the point of the
split. To add a skill you add one line to `tool_to_service` and one entry to
`TOOLS`; to change robot you replace those two things and nothing else.

```bash
export OPENAI_API_KEY=...        # never put a key in the source
python3 01_dry_run.py            # no ROS 2, no sim — just the loop

raise-sim                        # 02 needs the sim + the skill servers
ros2 run raise2026_tools navigation_server &
ros2 run raise2026_tools move_to_pose_server &
ros2 run raise2026_tools gripper_server &
ros2 run raise2026_tools inspector_server &
python3 02_raise_bot.py
```

**The dry run is not a toy.** Stubbing `call_service` is how you debug a system
prompt, a tool list and a mapping in seconds instead of minutes — the model, the
schemas and the loop are all real, only the robot is missing.

**`MAX_STEPS` is the guardrail.** The loop is bounded, so a confused agent stops
instead of running forever. Lesson 2.6 adds the rest.
