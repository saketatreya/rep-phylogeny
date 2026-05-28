# rep-phylogeny

**Can the composition error of Procrustes maps between language representations
recover a known phylogenetic tree?**

One model, five Romance languages, one number: the rank of the consensus
linguistic tree among all 15 possible unrooted binary topologies on 5 taxa.

If the correct topology lands at rank 1–2 on either of the two models we try,
the research direction is viable.

Full spec: [`experiment.md`](experiment.md).

---

## What's happening

1. **Extract** mean-pooled hidden states from a multilingual LM for the same
   1012 FLORES-200 sentences in each of 5 languages, at 4 layer depths.
2. **Fit** an orthogonal Procrustes map for each of the 10 language pairs on
   812 training sentences.
3. **Compose** maps two ways for each ordered triple `(A, B, C)` —
   direct `A→C` vs. indirect `A→B→C` — and measure the L2 mismatch on 200
   held-out test sentences. This yields 30 composition errors.
4. **Score** all 15 topologies. A topology predicts which intermediates
   should compose well (the ingroup) and which shouldn't (the outgroup).
   The topology whose ingroups give the lowest summed composition error
   wins.
5. **Compare** the winning topology to the consensus Romance tree.

| Models | Layers |
|---|---|
| `xlm-roberta-large` (560M, encoder) | 6, 12, 18, 23 |
| `google/gemma-2-2b` (2.6B, decoder, gated) | 6, 13, 19, 25 |

---

## Run on Kaggle (recommended)

1. Create a notebook with **GPU T4 x2** as the accelerator.
2. Open `kaggle_notebook.ipynb` from this repo (or copy the cells from it).
3. (For Gemma-2) Add an `HF_TOKEN` Kaggle Secret and accept the
   [Gemma-2 license](https://huggingface.co/google/gemma-2-2b).
4. Run all cells.

Total runtime: **≈ 10–15 minutes** on T4.

Combined report written to `/kaggle/working/results/report.txt`.

## Run locally

```bash
pip install -r requirements.txt
python run.py                          # both models, 4 layers each
python run.py --models xlm-roberta-large
python run.py --skip-extract           # reuse cached .npy reps
```

## Verify the math

```bash
python tests/test_pipeline.py
```

7 synthetic checks with no GPU/model required. With the planted
Romance-like structure, the ground-truth topology lands at rank ≤ 2.

---

## Layout

```
src/
  config.py            languages, layer indices, model registry
  data.py              FLORES-200 loader (3-source fallback)
  representations.py   model forward pass, layer-wise mean pooling
  procrustes.py        orthogonal Procrustes + transform
  composition.py       30 composition errors
  topologies.py        15 unrooted topologies + scoring
  report.py            ranking table + sanity checks
run.py                 CLI driver
tests/test_pipeline.py synthetic smoke test
kaggle_notebook.ipynb  Kaggle entry point
experiment.md          full spec
```

## Decision criteria

| Outcome | Verdict |
|---|---|
| Ground-truth rank 1 in any config | Strong signal — proceed |
| Ground-truth rank 1–3 in most configs | Viable |
| Ground-truth rank 4–7 consistently | Ambiguous |
| Ground-truth rank 8+ everywhere | Not viable — stop |

---

## Notes on spec resolution

A few things in `experiment.md` were ambiguous or wrong; resolved here as:

- **`itertools-more`** in the pip install — not a real package; dropped
  (only stdlib `itertools.combinations` is used).
- **FLORES dataset path** — `facebook/flores` is not always available;
  loader falls back across `facebook/flores → openlanguagedata/flores_plus →
  Muennighoff/flores200`.
- **Gemma-2 auth** — the spec doesn't mention that Gemma-2-2b is gated;
  it requires an HF token with the license accepted.
- **Rooted vs. unrooted outgroups** — the spec's outgroup table (lines
  430–441 of `experiment.md`) reflects the *rooted* tree
  `((((Spa,Por),Fre),Ita),Ron)`. When unrooted into the two splits
  `{0,1}|{2,3,4}` and `{0,1,2}|{3,4}`, the resulting tree has `(Ita,Ron)`
  as a cherry, so the split-based per-triple outgroup disagrees with the
  rooted-view outgroup on triples that cross the central edge. Since the
  algorithm scores **unrooted** topologies, it uses the split-based
  outgroup (matching the spec's `get_outgroup` function); the ground-truth
  topology is identified by its splits, not the rooted outgroup table.
