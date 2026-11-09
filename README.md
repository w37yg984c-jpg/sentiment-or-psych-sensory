# Sensory vs Psychological Description Classifier

## Research question
Can a logistic regression classifier distinguish sentences that describe
**sensory experience** (things perceivable through the five senses, body
movement, physical states) from sentences that describe **psychological
experience** (thoughts, emotions, judgments, memories, intentions)?

## Labels
- `0` — **Sensory**: the sentence mainly describes something perceivable
  through the senses, a body movement, or a physical state.
  e.g. "雨点敲打着窗户。" / "她穿着一件红色的外套。"
- `1` — **Psychological**: the sentence mainly describes a character's
  inner life — thoughts, emotions, judgments, memories, intentions.
  e.g. "她后悔了。" / "他觉得这件事很荒唐。"
- `-1` — ambiguous / neither; excluded from training.

### Unit
One sentence, ending with 。！？ (8–60 characters after cleaning).

### Boundary rules
1. If a sentence contains both, label by the **main predicate / main object
   of description**: external action/object → 0; inner activity → 1.
2. Direct speech: label by the **content** of the quoted speech; if the
   sentence only reports a speech act, label `-1`.
3. Pure scenery/environment with no person → 0.
4. Abstract general statements not tied to a character's mind → `-1`.
5. When in doubt, label `-1` (skip). Target: ≥200 items per class.

## Data
- Source: Zhang Ailing (Eileen Chang), *Half a Lifetime of Fate* (《半生缘》).
- Used for coursework only; the full text is **not** committed to this
  repository (`data/raw/` is git-ignored). Only labelled sentences are shared.
- `data/train.csv` — final labelled data (`text,label`).
- `data/to_label.csv` — raw sampled sentences before labelling.
- `data/ai_labeled.csv` — AI-assisted labels before human spot-check.
- `data/spotcheck.csv` — 50-item human spot-check of AI labels.

## Labelling procedure
1. Split the source text into sentences (`src/split_sentences.py`).
2. AI-assisted pre-labelling via OpenRouter (`src/label_with_ai.py`).
   The API key is read from the environment variable `OPENROUTER_API_KEY`
   and is never stored in this repository.
3. Human spot-check of 50 randomly sampled items; disagreements corrected.
4. Final labels saved to `data/train.csv`.

## How to run
```bash
pip install -r requirements.txt
python src/split_sentences.py     # build data/to_label.csv
export OPENROUTER_API_KEY=...     # your own key; never commit it
python src/label_with_ai.py       # AI pre-labels -> data/ai_labeled.csv
# (human spot-check -> data/train.csv)
python src/train.py               # train + evaluate -> output/
python src/apply.py               # predict on analysis text -> output/