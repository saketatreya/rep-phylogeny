# Genealogy over surface

Spec: [`new_spec.md`](new_spec.md). Three escalating gates:

- **Tier 0:** Atomic flip on {English, German, French}. One sign.
- **Tier 1:** Held-out family discriminant. English overrides its surface
  similarity to French and classifies as Germanic.
- **Tier 2:** Single inherited-vs-borrowed axis transferring across families
  (English ↔ Germanic, Maltese ↔ Semitic, Urdu ↔ Indo-Aryan, Romanian ↔
  Romance).

## Layout

```
src/
  config.py         language set (25 langs), model + sweep grid
  data.py           FLORES-200 tarball loader (generic over codes)
  representations.py extraction: mean-pool + high-frequency-token pool
  surface.py        char n-gram TF-IDF distance (with romanization helper)
  hub.py            generalized orthogonal Procrustes
  geometry.py       Procrustes residual + hub centroid distances
  classify.py       LDA family discriminant + axis projection
  tree.py           NJ + Newick + bootstrap clade support + Mantel
  bootstrap.py      test-sentence resampling helpers
run_extract.py      GPU stage — one-shot per model
run_tier0.py        atomic flip table
run_tier1.py        held-out discriminant
run_tier2.py        full sweep (clade support, axis transfer, Mantel, translit)
kaggle_notebook.ipynb  orchestrates extraction + all tiers
```

## Run

**Kaggle (GPU):** open `kaggle_notebook.ipynb`, run all. Outputs land in
`/kaggle/working/results/`; the final cell zips them.

**Local CPU (Tier 0 only, after extraction exists locally):**
```
python run_tier0.py --reps-dir results/reps --out-dir results
```

Bootstrap and clade-support stages need `results/reps/` populated by
`run_extract.py`. Local extraction with all 25 languages and 24 layers is
feasible on CPU but slow (~hours); use Kaggle for the full sweep.

## Outputs

Read `results/SUMMARY.txt` top-to-bottom — it's the decision table.
Per-tier CSVs (`tier0_atomic.csv`, `tier1_heldout.csv`,
`tier2a_clade_support_*.csv`, …) carry the full numbers. NJ trees go to
`results/trees/*.nwk`.
