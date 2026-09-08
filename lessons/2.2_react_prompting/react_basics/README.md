# Lesson 2.2 — the ReAct loop

The same loop twice: once on toy functions, once on the real robot.

| file | tools | what it shows |
|---|---|---|
| `01_react_toy.py` | two Python functions | Thought → Action → Observation, printed every turn |
| `02_react_robot.py` | three ROS 2 Trigger services | the identical loop, driving the greenhouse robot |

The only change between them is the `SKILLS` table. The loop, the schemas and
the prompt are the same — which is the point: ReAct does not care what is behind
the tool.

```bash
export OPENAI_API_KEY=...        # never put a key in the source
python3 01_react_toy.py          # no robot needed

raise-sim                        # 02 needs the sim + the skill servers
ros2 run raise2026_tools navigation_server &
ros2 run raise2026_tools move_to_pose_server &
ros2 run raise2026_tools inspector_server &
python3 02_react_robot.py
```

**How the reasoning becomes visible:** every tool schema carries one extra
argument, `thought`. The model must fill it in before it is allowed to act, so
the trace it prints is its real reasoning — not a summary written afterwards.
