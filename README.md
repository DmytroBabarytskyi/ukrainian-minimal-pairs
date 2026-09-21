# Ukrainian Minimal Pairs: Knowing the Norm vs. Using the Norm

A benchmark of **1,444 Ukrainian minimal pairs** across six grammatical and
lexical phenomena, with a scoring harness that evaluates a language model in
two ways: by comparing sentence probabilities, and by asking the model directly
which sentence is correct.

Every pair differs in exactly **one edited span**: one word, one form, or one
construction replaced by another. One member conforms to the norms of standard
Ukrainian; the other does not. In 44 pairs the two members have different word
counts, because the construction that changes has a different length on each
side («протягом року» / «на протязі року»); `word_diff` and `len_equal` record
this for every pair.

```json
{"uid": "vocative_0001", "phenomenon": "vocative", "group": "vocative m",
 "sentence_good": "Олександре, подивись сюди.",
 "sentence_bad":  "Олександр, подивись сюди.",
 "word_diff": 1, "len_equal": true, "source": "template+curated names"}
```

## Why this exists

Minimal-pair benchmarks are well established for English (BLiMP) and Russian
(RuBLiMP). For Ukrainian, MultiBLiMP 1.0 covers subject–verb agreement only.
This set adds the vocative case, numeral–noun agreement, adjective agreement,
verb government, and calques from Russian — the last being a phenomenon that
English-language benchmarks cannot have by construction: forms that are fluent,
frequent, and non-normative.

Subject–verb agreement is included as a **control**: it overlaps with
MultiBLiMP, so a harness that works should score near ceiling on it.

## Contents

| Path | What |
|---|---|
| `pairs/*.jsonl` | the pairs, one JSON object per line, grouped by phenomenon |
| `data/*.tsv` | curated source lists: names, calques, verb government |
| `gen.py` | generators that build the pairs from templates and lists |
| `evaluate.py` | scoring harness, one CSV per model |
| `kaggle_run.py` | sweep driver for Kaggle (resumable across sessions) |
| `analysis/` | aggregation, validation statistics, figures, paper tables |
| `results/*.csv` | raw per-pair scores for every model evaluated |
| `validation/` | human validation sheets and answer key |

## Phenomena

| Phenomenon | Pairs | Example (✓ / ✗) |
|---|---|---|
| Subject–verb agreement (control) | 300 | вона прийшла / вона прийшов |
| Adjective agreement | 300 | нова книжка лежить / новий книжка лежить |
| Numeral + noun | 300 | два столи / два стола |
| Vocative case | 300 | Олеже, зайди / Олег, зайди |
| Calques from Russian | 184 | брати участь / приймати участь |
| Verb government | 60 | дякую вам / дякую вас |

Normativity follows Ukrainian reference works: Serbenska, *Antysurzhyk*;
Ponomariv, *Kultura slova*; Ukrainian Orthography (2019). Items on which those
sources disagree, or which are contested among native speakers, were excluded
rather than adjudicated.

## Human validation

A 368-pair sample (25% of the set) was rated independently by two native
speakers with no access to the key and no dictionaries.

| Metric | Value |
|---|---|
| Cohen's κ | 0.963 |
| Raw agreement | 98.1% |
| Defect rate (both raters against the key) | 1.1% |

One group (`subtle: ймовірно`, «ймовірно» / «вірогідно») was rejected by both
raters as a distinction of nuance rather than of normativity, and the **whole
group** was dropped — not only the rated items. That is why the released set
has 1,444 pairs and not 1,448.

Reproduce with `python analysis/validation.py`.

## Usage

```bash
git clone https://github.com/DmytroBabarytskyi/ukrainian-minimal-pairs
cd ukrainian-minimal-pairs
```

Score one model:

```bash
python evaluate.py --model google/gemma-3-4b-it --4bit
```

Useful flags: `--dtype fp32` (Gemma 3 at 12B overflows fp16 on a T4 and returns
NaN), `--drop-vision` (frees the image encoder of a multimodal checkpoint,
unused for text scoring), `--max-memory-gb` (offload to CPU), `--limit` (smoke
test). Output is `results/<model>.csv`, one row per pair.

Then aggregate:

```bash
python analysis/analyze.py && python analysis/figures.py
```

### Scoring

**Probability.** Sum of token log-probabilities per sentence; a pair counts as
correct when `logP(good) > logP(bad)`. Token counts are stored so that a
length-normalised variant can be reported alongside — this matters, because some
phenomena change the token count of the sentence.

**Prompt.** The model is asked which sentence is correct and the
log-probabilities of answering "А" and "Б" are compared. **Each pair is asked in
both orders.** The share of pairs answered the same way in both orders is
reported as order consistency; without it, a prompted accuracy figure cannot be
distinguished from a fixed preference for one answer position.

## Requirements

`torch`, `transformers`, `pandas`, `matplotlib`; `bitsandbytes` for 4-bit
loading, `openpyxl` to read the validation sheets, `pymorphy3` to regenerate the
pairs. Tested on a Kaggle T4.

## Licence

- **Data** (`pairs/`, `data/`, `validation/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `LICENSE-DATA`.
- **Code** (everything else): MIT — see `LICENSE`.

## Citation

See `CITATION.cff`. A paper describing the set is in preparation; this section
will carry the reference once it has a venue.
