"""Lesson 3.7 - step 1: a word is already a number.

WHAT  Run a real language-model tokenizer over the instruction our robot is
      given, and print the numbers it turns into.

WHY   Everyone accepts that an LLM "predicts the next word".  What it actually
      predicts is an INDEX into a fixed vocabulary - a plain integer.  Hold on
      to that: in three scripts we will do the very same thing to a joint angle.

    python3 01_a_word_is_a_number.py
"""

from transformers import AutoTokenizer

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'     # any LLM tokenizer makes the point
INSTRUCTION = 'pick the red tomato'      # the task string in our dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL)
ids = tokenizer.encode(INSTRUCTION, add_special_tokens=False)
words = tokenizer.convert_ids_to_tokens(ids)

print(f'tokenizer   {MODEL}     vocabulary: {tokenizer.vocab_size:,} words')
print(f'instruction "{INSTRUCTION}"')
print()
for word, i in zip(words, ids):
    # 'G-dot' in front of a token is this tokenizer's way of writing a space.
    print(f'   {word.replace(chr(288), "_"):<10s} ->  {i:6d}')
print()
print(f'the model never sees the words: it sees {ids}')
