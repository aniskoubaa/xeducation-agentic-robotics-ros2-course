# Author: Prof. Anis Koubaa <anis.koubaa@gmail.com>
"""
RAISE 2026 — the shared sim joint poses for Day-2 manipulation.

ONE place for the tuned UR5e poses that the auto-demonstrator (05) records
with and the Lab-2.2 evaluator tests with. They MUST match: the evaluator
spawns tomatoes at the same reachable grasp points the demos were collected
at, so the policy is tested on the distribution it was trained on.

Values are joint targets in radians, UR5E_JOINTS order:
    [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]

GEOMETRY (settled after live iteration — 2026-07-05, the "camera must see the
task" lesson): the camera rides the wrist, so WHAT THE MODEL CAN KNOW is set
by these poses.
- Downward "bird's-eye" family (this file): at above()/HOME the camera looks
  down and clearly sees the tomato under it, with the PLANT POT + FOLIAGE in
  frame (the robot is parked at a plant row — see PARK).
- The demo choreography is a SCAN: look above LEFT → if the tomato there is
  red, descend; else pan to RIGHT and descend there. Every frame is
  unambiguous ("red below me → go down"), so the policy can decide from the
  CURRENT image alone. A horizontal-approach variant looked prettier but hid
  the tomatoes behind the gripper — the model plateaued at 50% (chance).

Robot parking (poses are tuned for THIS spot — re-park ⇒ re-record):

    raise-sim x:=-2.0 y:=2.15 yaw:=1.5708     # facing the plant at (-2, 3)
"""

POSE_HOME = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
GRASP_LEFT = [0.35, -0.70, 1.30, -2.00, -1.5708, 0.0]
GRASP_RIGHT = [-0.35, -0.70, 1.30, -2.00, -1.5708, 0.0]

# The parking pose the poses above were tuned for (documented + used by demos).
PARK = {'x': -2.0, 'y': 2.15, 'yaw': 1.5708}


def above(pose, lift=0.45):
    """The 'look/approach' pose above a grasp pose (shoulder_lift more negative).

    At above(GRASP_x) the camera sees that side's tomato clearly (verified by
    live captures) — this is the SCAN viewpoint the choreography decides from.
    """
    p = list(pose)
    p[1] -= lift
    return p
