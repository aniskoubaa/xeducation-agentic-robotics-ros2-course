# action_words_demo — how a model outputs a movement

Lesson 3.7 asks the question that puzzles most newcomers: *how can a neural
network, which only predicts numbers, produce a physical movement?* Slide 3's
answer is one sentence — **predicting an action is just like predicting a
word** — and these six steps make that sentence literal, on the dataset the
previous lessons already opened (`raiseschool/raise_ripeness_sort_ref`).

**No model is loaded and no GPU is touched.** Lesson 3.10 picks a VLA and 3.11
runs one; this lesson is about the *representation*, which you can build and
check yourself in a few lines.

| file | it shows |
|---|---|
| `codebook.py` | the action vocabulary: OpenVLA's 256-bin recipe, in four functions |
| `01_a_word_is_a_number.py` | a real LLM tokenizer on our instruction — words are indices |
| `02_the_codebook.py` | build the vocabulary from our own actions; how wide is one bin? |
| `03_an_action_is_a_number.py` | one real action becomes seven integers |
| `04_what_binning_costs.py` | encode all 4228 actions, decode them back, measure the damage |
| `05_side_by_side.py` | the sentence and the movement, in the same layout |
| `06_two_ways_out.py` | discrete action words vs a continuous action expert |

## Run them

LeRobot and `transformers` live in the same virtualenv:

```bash
ln -sfn ../../../datasets/raiseschool__raise_ripeness_sort_ref dataset
export HF_HUB_OFFLINE=1                   # the tokenizer is already cached

~/raise_venvs/lerobot/bin/python3 01_a_word_is_a_number.py
~/raise_venvs/lerobot/bin/python3 02_the_codebook.py
~/raise_venvs/lerobot/bin/python3 03_an_action_is_a_number.py
~/raise_venvs/lerobot/bin/python3 04_what_binning_costs.py
~/raise_venvs/lerobot/bin/python3 05_side_by_side.py
~/raise_venvs/lerobot/bin/python3 06_two_ways_out.py
```

`dataset/` is gitignored. `01` and `05` need a cached LLM tokenizer
(`Qwen/Qwen2.5-1.5B-Instruct`); any tokenizer makes the same point.

## Three things worth knowing

**256 bins costs almost nothing.** The six arm joints survive the round trip
through the vocabulary with an average error of **0.030 degrees** — well below
what the arm can repeat mechanically. That is the entire reason the discrete
trick is allowed to work: you can treat a smooth movement as a sequence of
choices without the robot noticing.

**The gripper only ever uses two of its 256 words.** Open and closed, and
nothing between — the same fact lesson 3.6 measured from the other side. A
vocabulary can be far larger than the thing you are describing needs.

**Do not use `meta/stats.json`'s `q01`/`q99` for the bin edges.** LeRobot
aggregates those per episode, so for a channel whose value differs from episode
to episode — `shoulder_pan`, which depends on which side the tomato is — the
stored range is far too narrow: it clips **36%** of the real values and throws
away up to 19 degrees. `codebook.py` computes the percentiles across the whole
table instead, which is what OpenVLA actually does, and clipping drops to ~2%.
