# Lesson 3.12 — Evaluating the Fine-Tuned Policy

Lesson 3.11 trained SmolVLA on **all 49** episodes, which left nothing to test
on. So we trained it again — same 20,000 steps, same batch 16, same seed — on
**episodes 0–39**, and kept **40–48** back. `model/` is that checkpoint;
`model_all/` is 3.11's, kept only so the two episode lists can be shown side by
side.

| file | what it shows |
|---|---|
| `smol.py` | the split, the shared names, and `quiet()` — SmolVLA chatters as it loads |
| `01_the_split.py` | the episode list read straight out of each checkpoint's own record |
| `02_the_yardstick.py` | how far apart two human demonstrations of the same scene are |
| `03_run_the_eval.py` | the sweep: 311 predictions - held-out, trained-on, and the base model |
| `04_the_scoreboard.py` | the three-row table, and `out/by_joint.png` |
| `05_the_loss_is_not_the_answer.py` | every checkpoint scored against the loss, `out/loss_vs_error.png` |
| `06_the_mean_lies.py` | percentiles, where in the episode the error sits, `out/where_the_error_is.png` |
| `07_the_closest_thing_to_a_rollout.py` | nine episodes replayed at 3.8's rhythm, `out/gripper.png` |
| `08_what_it_does_not_prove.py` | the nearest-neighbour baseline, and the online checklist |

## Run them

```bash
python3 01_the_split.py
python3 02_the_yardstick.py
python3 03_run_the_eval.py         # ~50 s on one GPU, writes out/eval.npz
python3 04_the_scoreboard.py       # writes out/by_joint.png
python3 05_the_loss_is_not_the_answer.py       # ~45 s, writes out/loss_vs_error.png
python3 06_the_mean_lies.py        # writes out/where_the_error_is.png
python3 07_the_closest_thing_to_a_rollout.py   # writes out/gripper.png
python3 08_what_it_does_not_prove.py
```

Scripts 4, 6 and 8 read `out/eval.npz`, so run `03` first. `05` needs
`checkpoints/` — a link to the run's own checkpoint folder — and
`runs/split_20000.log`, the step log that run printed.

## The offline metric, in full

```python
chunk = post(policy.predict_action_chunk(pre(observation)))   # (1, 50, 7)
error = numpy.degrees(numpy.abs(chunk[0] - human_actions))    # (50, 7), in degrees
```

That is all an action error is: the plan minus what the human commanded over
the same five seconds. Everything else in this demo is a way of *reading* it —
by joint, by percentile, by moment in the episode, by how far ahead the policy
committed.

## What you should take away

SmolVLA's flow matching starts from noise, so every number below moves in the
last decimal from run to run. The values quoted are from one filmed take.

* **The loss curve is not the answer.** It falls smoothly to **0.024** and never
  says when to stop. After step 5,000 the loss fell another **78 %** while the
  held-out error fell only **25 %** — and the last 5,000 steps moved that error
  by as much as simply measuring it twice does.
* **Hold data back, or you learn nothing.** Held-out scores **0.54 deg**;
  episodes the policy trained on score **0.20**. That **+0.34** gap — about
  three times worse on data it never saw — is the memorisation test, and it is
  invisible if you train on everything, which is exactly what lesson 3.11 did.
* **One number is never enough.** 0.54 means nothing until you put it beside the
  base model's **35**, and the **0.35** that separates two human demonstrations.
* **The mean hides the moments that matter.** Half the steps are under 0.11 deg,
  but the worst is **12.2** — and the error peaks at the same instant, **2.0 s
  in, in *both* scenes**: 0.97 deg against 0.39 everywhere else, and well away
  from the grasp.
* **Count events, don't only average them.** The gripper closed in **9 of 9**
  held-out episodes, within one robot step of the human, with no spurious
  closes. That is worth more than any average.
* **Compare against something trivial.** A lookup table returning the nearest
  training episode scores **0.18 deg** — better than the policy's 0.54. Not because the
  policy is bad, but because this held-out set is too easy to be evidence.

## What this is *not*

Every number here is **offline**. The policy is fed recorded observations, so
these scripts measure whether it reproduces demonstrations — never whether the
robot picks the tomato. A success rate needs the robot, `N` trials at **new**
tomato positions and **new** lighting, and the wrong (green) grabs counted
separately as a safety number. The loss curve is not the answer; neither is
this table.
