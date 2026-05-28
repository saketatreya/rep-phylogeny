# Phylogenetic Tree Recovery — Viability Experiment Implementation Spec

## Objective

Determine whether the composition error of linear transformations between language representations in a multilingual model recovers the known phylogenetic tree for the Romance language family. One model, five languages, one number: the rank of the correct tree topology among all 15 possible topologies on 5 taxa. If the correct topology is rank 1 or 2, the research direction is viable.

---

## Infrastructure

**Platform:** Kaggle, 2× NVIDIA T4 (16 GB VRAM each)

**Python packages to install:**

```bash
pip install torch transformers datasets scipy numpy itertools-more tqdm
```

**Models (run one per GPU in parallel if desired):**

| Model | HuggingFace ID | Params | fp16 VRAM | Type |
|-------|---------------|--------|-----------|------|
| XLM-RoBERTa-large | `xlm-roberta-large` | 560M | ~2.2 GB | Encoder (bidirectional) |
| Gemma-2-2B | `google/gemma-2-2b` | 2.6B | ~5.2 GB | Decoder (autoregressive) |

Both fit comfortably on a single T4 in fp16. Run both — it costs almost nothing extra and gives two independent data points. If you must pick one, start with XLM-R (purpose-built for cross-lingual work, trivially small).

---

## Data

**Source:** FLORES-200, available via HuggingFace.

```python
from datasets import load_dataset
flores = load_dataset("facebook/flores", "all_languages")
# Use the "devtest" split — 1012 parallel sentences
```

**Languages and FLORES codes:**

| Index | Language | FLORES column |
|-------|----------|---------------|
| 0 | Spanish | `sentence_spa_Latn` |
| 1 | Portuguese | `sentence_por_Latn` |
| 2 | French | `sentence_fra_Latn` |
| 3 | Italian | `sentence_ita_Latn` |
| 4 | Romanian | `sentence_ron_Latn` |

The index assignments (0–4) are used throughout this document and must be consistent in the code. Name the languages in all output by their full names, not indices.

**Sentence count:** 1012 sentences in devtest. All 1012 sentences are parallel across all 5 languages — same meaning, same order.

---

## Pipeline Overview

```
Step 1: Extract representations  →  5 matrices, one per language, shape (1012, d_model)
Step 2: Split data               →  train (812 sentences), test (200 sentences)
Step 3: Fit Procrustes maps      →  10 orthogonal matrices, one per language pair
Step 4: Compute composition errors →  30 scalar values (10 triples × 3 intermediates each)
Step 5: Score all 15 topologies  →  ranked list of topologies by total composition score
Step 6: Compare to ground truth  →  report rank of the consensus Romance tree
```

---

## Step 1: Extract Representations

For each language, run all 1012 sentences through the model. Extract hidden states. Mean-pool over token positions (excluding special tokens) to produce one vector per sentence.

**Extract at 4 layers** for each model:

| Label | Layer index (0-based) |
|-------|-----------------------|
| L_quarter | `n_layers // 4` |
| L_half | `n_layers // 2` |
| L_three_quarter | `3 * n_layers // 4` |
| L_final | `n_layers - 1` |

For XLM-R-large: 24 layers, d=1024. Layers 6, 12, 18, 23.
For Gemma-2-2B: 26 layers, d=2304. Layers 6, 13, 19, 25.

**Implementation notes:**

- Load model in fp16: `model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16)`
- Use `output_hidden_states=True` to get all layer outputs in a single forward pass
- Batch size: 16–32 should fit on a T4 for both models
- For XLM-R, exclude `[CLS]` (index 0) and `[SEP]` (last non-pad token) from the mean pool. Pool over the remaining token positions.
- For Gemma-2, exclude `<bos>` (index 0) and any padding tokens. Pool over the remaining positions.
- Convert pooled vectors to float32 numpy arrays before saving (linear algebra is in numpy, not torch)

**Output of Step 1:**

For each model and each layer, save 5 numpy arrays:

```
representations/{model_name}/{layer_label}/{lang}.npy  — shape (1012, d_model), dtype float32
```

**Sanity check:** For each language pair, compute the cosine similarity between corresponding sentence representations. Related languages (Spanish–Portuguese) should show higher mean similarity than distant pairs (Spanish–Romanian). If all pairs show near-identical similarity, something is wrong (likely pooling over padding or including CLS token which encodes language identity).

---

## Step 2: Split Data

Split the 1012 sentences into:

- **Train:** sentences 0–811 (812 sentences) — used to fit Procrustes maps
- **Test:** sentences 812–1011 (200 sentences) — used to evaluate composition error

This split is fixed and identical across all pairs. No randomness.

**Why a simple split, not disjoint per-pair splits:** For the viability check, a shared train split is fine. If the correct topology emerges despite shared training data (which makes composition trivially easier), the signal is real. Strict disjoint splits are for the paper, not the viability gate. If viability is confirmed and you want to tighten this, split the 812 training sentences into 4 disjoint blocks of 203 sentences each, fit each pair on a different block, and test on the 200-sentence test set. But do the simple version first.

---

## Step 3: Fit Procrustes Maps

For each of the 10 ordered pairs (A, B) where A < B, fit an orthogonal Procrustes transformation mapping language A representations to language B representations.

**The 10 pairs:**
(0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)

**Procrustes solution:**

Given `X_A` (n×d) and `X_B` (n×d) — the train-split representations for languages A and B:

```python
import numpy as np
from scipy.linalg import svd

def fit_procrustes(X_A, X_B):
    """Find orthogonal R minimizing ||X_A @ R - X_B||_F.
    
    Args:
        X_A: (n, d) source representations
        X_B: (n, d) target representations
    
    Returns:
        R: (d, d) orthogonal matrix such that X_A @ R ≈ X_B
    """
    M = X_A.T @ X_B          # (d, d)
    U, S, Vt = svd(M)        # M = U @ diag(S) @ Vt
    R = U @ Vt                # orthogonal solution
    return R
```

**Center the data first.** Before fitting, subtract the mean of each language's training representations:

```python
mean_A = X_A_train.mean(axis=0)
mean_B = X_B_train.mean(axis=0)
X_A_centered = X_A_train - mean_A
X_B_centered = X_B_train - mean_B
R_AB = fit_procrustes(X_A_centered, X_B_centered)
```

Store the means alongside R for use at test time.

**Deriving reverse maps:** `R_BA = R_AB.T` (transpose of an orthogonal matrix is its inverse). Do not fit separately — this ensures consistency.

**Output of Step 3:**

A dictionary of 10 Procrustes maps plus their associated means:

```python
procrustes[(A, B)] = {
    'R': R_AB,        # (d, d) orthogonal matrix
    'mean_A': mean_A, # (d,) mean of A's training representations
    'mean_B': mean_B  # (d,) mean of B's training representations
}
```

**Sanity check:** Compute reconstruction error on the test set for each pair: `||X_A_test_centered @ R_AB - X_B_test_centered||_F / n_test`. Closer languages (Spanish–Portuguese) should have lower reconstruction error than distant ones (Spanish–Romanian).

---

## Step 4: Compute Composition Errors

For each ordered triple (A, B, C), where B is the intermediate, compute:

**composition error = mean over test sentences of** `|| transform(A→B→C, x_A) - transform(A→C, x_A) ||₂`

Concretely, for a test sentence with representations `x_A`, `x_B`, `x_C`:

```
direct:   x_A_pred_C = (x_A - mean_A) @ R_AC + mean_C     # direct A→C
indirect: x_A_pred_B = (x_A - mean_A) @ R_AB + mean_B     # first hop A→B
          x_A_pred_C_via_B = (x_A_pred_B - mean_B) @ R_BC + mean_C  # second hop B→C
composition_error = || x_A_pred_C_via_B - x_A_pred_C ||₂
```

Where `R_AC` and `R_AB` and `R_BC` may use transposes depending on direction:
- If A < C, use `R_AC` directly and `mean_A`, `mean_C` as stored
- If A > C, use `R_CA.T` and swap means accordingly

**Simplification:** Since centering and re-centering with the same mean cancels out, the indirect path simplifies:

```python
def apply_transform(x, pair, procrustes_dict, direction='forward'):
    """Apply Procrustes transform to a single point or batch.
    
    direction='forward': A→B using stored R
    direction='reverse': B→A using R.T
    """
    A, B = pair
    R = procrustes_dict[(A,B)]['R']
    mean_A = procrustes_dict[(A,B)]['mean_A']
    mean_B = procrustes_dict[(A,B)]['mean_B']
    
    if direction == 'forward':
        return (x - mean_A) @ R + mean_B
    else:  # reverse: B→A
        return (x - mean_B) @ R.T + mean_A

def get_transform(src, dst, procrustes_dict):
    """Return a function that maps src representations to dst."""
    if (src, dst) in procrustes_dict:
        return lambda x: apply_transform(x, (src, dst), procrustes_dict, 'forward')
    elif (dst, src) in procrustes_dict:
        return lambda x: apply_transform(x, (dst, src), procrustes_dict, 'reverse')
    else:
        raise KeyError(f"No Procrustes map for ({src}, {dst})")
```

**Compute all 30 composition errors:**

There are C(5,3) = 10 unordered triples. For each triple {A, B, C}, there are 3 choices of intermediate: A, B, or C. That gives 30 composition error values. However, we only need to compute the unique directed triples where the intermediate is each possible language.

```python
from itertools import combinations

comp_errors = {}  # (A, intermediate, C) → scalar error

for triple in combinations(range(5), 3):
    for intermediate in triple:
        others = [x for x in triple if x != intermediate]
        src, dst = others[0], others[1]
        
        T_src_to_int = get_transform(src, intermediate, procrustes_dict)
        T_int_to_dst = get_transform(intermediate, dst, procrustes_dict)
        T_src_to_dst = get_transform(src, dst, procrustes_dict)
        
        # Apply to test data
        x_src = X_test[src]  # (n_test, d)
        indirect = T_int_to_dst(T_src_to_int(x_src))
        direct = T_src_to_dst(x_src)
        
        # Mean L2 error per sentence
        error = np.mean(np.linalg.norm(indirect - direct, axis=1))
        comp_errors[(src, intermediate, dst)] = error
        
        # Also compute the reverse direction
        src2, dst2 = others[1], others[0]
        T_src2_to_int = get_transform(src2, intermediate, procrustes_dict)
        T_int_to_dst2 = get_transform(intermediate, dst2, procrustes_dict)
        T_src2_to_dst2 = get_transform(src2, dst2, procrustes_dict)
        
        indirect2 = T_int_to_dst2(T_src2_to_int(X_test[src2]))
        direct2 = T_src2_to_dst2(X_test[src2])
        error2 = np.mean(np.linalg.norm(indirect2 - direct2, axis=1))
        comp_errors[(src2, intermediate, dst2)] = error2
```

**Note:** For each unordered triple {X,Y,Z} with intermediate Y, there are two directed triples: (X,Y,Z) and (Z,Y,X). Average them to get a single composition error per intermediate:

```python
# For triple {X,Y,Z} with intermediate Y:
ce_Y = 0.5 * (comp_errors[(X, Y, Z)] + comp_errors[(Z, Y, X)])
```

This yields 30 values total, or equivalently, 10 triples × 3 intermediates = 30 averaged composition errors.

**Output of Step 4:** A table with 10 rows (one per triple) and 3 columns (one per intermediate choice), containing composition error values.

---

## Step 5: Score All 15 Topologies

### 5a. Enumerate topologies

An unrooted binary tree on 5 labeled leaves has exactly 15 possible topologies. Each topology is fully characterized by its set of 2 non-trivial bipartitions (splits) of the 5 taxa.

A non-trivial split partitions {0,1,2,3,4} into a group of 2 and a group of 3. There are C(5,2) = 10 possible splits. Two splits are **compatible** if, when we write them as (A|B) and (C|D), at least one of the four intersections A∩C, A∩D, B∩C, B∩D is empty.

Each topology = a pair of compatible splits.

```python
from itertools import combinations

def enumerate_splits(n=5):
    """All non-trivial splits of {0,...,n-1} into groups of size 2 and 3."""
    taxa = set(range(n))
    splits = []
    for pair in combinations(range(n), 2):
        small = frozenset(pair)
        big = taxa - small
        splits.append((small, big))
    return splits

def are_compatible(s1, s2):
    """Two splits are compatible if one of the four intersections is empty."""
    a1, b1 = s1
    a2, b2 = s2
    return (not a1 & a2) or (not a1 & b2) or (not b1 & a2) or (not b1 & b2)

def enumerate_topologies():
    """Return all 15 unrooted binary trees on 5 labeled taxa.
    Each topology is a tuple of 2 compatible splits.
    """
    splits = enumerate_splits()
    topologies = []
    for i in range(len(splits)):
        for j in range(i+1, len(splits)):
            if are_compatible(splits[i], splits[j]):
                topologies.append((splits[i], splits[j]))
    assert len(topologies) == 15, f"Expected 15, got {len(topologies)}"
    return topologies
```

### 5b. Determine outgroups from topology

For each topology and each triple of taxa, the topology determines which taxon is the **outgroup** (branches off first from the other two).

```python
def get_outgroup(topology, triple):
    """Given a topology (pair of splits) and a triple of taxa,
    determine which taxon is the outgroup.
    
    The outgroup is the taxon that is separated from the other two
    by one of the topology's splits.
    
    Returns the outgroup taxon, or None if the topology doesn't 
    resolve this triple (shouldn't happen for binary trees).
    """
    triple_set = set(triple)
    for split in topology:
        small, big = split
        # Check if the split separates one member of the triple from the other two
        in_small = triple_set & small
        in_big = triple_set & big
        if len(in_small) == 1 and len(in_big) == 2:
            return list(in_small)[0]
        if len(in_big) == 1 and len(in_small) == 2:
            return list(in_big)[0]
    # If neither split resolves the triple, all three are on the same side
    # of both splits. The outgroup is the one not in either cherry.
    # This requires more careful handling:
    cherries = [split[0] for split in topology]  # the size-2 groups
    for taxon in triple:
        others = triple_set - {taxon}
        if others in [frozenset(c) for c in cherries]:
            return taxon  # the other two form a cherry, this one is the outgroup
    # Fallback: no clean resolution. Should not happen for resolved binary trees.
    return None
```

### 5c. Score each topology

**Scoring method (minimize):**

For each topology and each of the 10 triples {X,Y,Z}:

1. The topology predicts an outgroup O. The other two are the ingroup I1, I2.
2. Look up the composition error when using each taxon as intermediate.
3. The topology's score for this triple = composition error using I1 as intermediate + composition error using I2 as intermediate. (These are the "good" compositions that the topology predicts should have low error.)

Total topology score = sum over all 10 triples.

**The topology with the lowest total score wins** — it correctly predicts which compositions are easy (through closely related intermediates).

```python
def score_topology(topology, comp_error_table):
    """Score a topology by total composition error through non-outgroup intermediates.
    
    Args:
        topology: pair of splits
        comp_error_table: dict mapping (triple_as_frozenset, intermediate) → error
    
    Returns:
        total score (lower is better)
    """
    total = 0.0
    for triple in combinations(range(5), 3):
        outgroup = get_outgroup(topology, triple)
        ingroup = [t for t in triple if t != outgroup]
        # Sum composition errors through both ingroup intermediates
        for intermediate in ingroup:
            total += comp_error_table[frozenset(triple), intermediate]
    return total
```

**Alternative scoring method (vote count — use as a secondary diagnostic):**

For each of the 10 triples, identify which intermediate empirically has the **highest** composition error (the empirical outgroup). For each topology, count how many triples have the topology's predicted outgroup matching the empirical outgroup. Best topology = highest count (out of 10). This is more interpretable and serves as a sanity check.

**Implement both scoring methods.** They should agree if the signal is clean.

---

## Step 6: Compare to Ground Truth

### The consensus Romance tree

The standard phylogeny for these 5 languages:

```
Romanian (4) branches off first          — Eastern Romance
  then Italian (3) branches off          — Italo-Dalmatian
    then French (2) branches off         — Gallo-Romance
      then Spanish (0) and Portuguese (1) split — Ibero-Romance
```

As an unrooted binary tree, this corresponds to 2 splits:

- **Split 1:** {0, 1} | {2, 3, 4}  →  (Spanish, Portuguese) vs (French, Italian, Romanian)
- **Split 2:** {0, 1, 2} | {3, 4}  →  (Spanish, Portuguese, French) vs (Italian, Romanian)

**Wait — check this.** The rooted tree is `((((Spanish, Portuguese), French), Italian), Romanian)`. Converting to unrooted splits by cutting each internal edge:

- Cut between (Spa,Por) cherry and French: yields {0,1} | {2,3,4} ✓
- Cut between (Spa,Por,Fra) clade and Italian: yields {0,1,2} | {3,4} ✓

**So the ground truth topology = `({0,1}, {2,3,4}), ({0,1,2}, {3,4})`**

Equivalently, verify using outgroup predictions for each triple:

| Triple | Outgroup | Reasoning |
|--------|----------|-----------|
| {Spa, Por, Fre} | Fre (2) | Spa-Por are the cherry |
| {Spa, Por, Ita} | Ita (3) | Spa-Por are the cherry |
| {Spa, Por, Ron} | Ron (4) | Spa-Por are the cherry |
| {Spa, Fre, Ita} | Ita (3) | Spa-Fre are closer (both Western) |
| {Spa, Fre, Ron} | Ron (4) | Spa-Fre are closer |
| {Spa, Ita, Ron} | Ron (4) | Spa-Ita closer than Spa-Ron |
| {Por, Fre, Ita} | Ita (3) | Por-Fre are closer (both Western) |
| {Por, Fre, Ron} | Ron (4) | Por-Fre are closer |
| {Por, Ita, Ron} | Ron (4) | Por-Ita closer than Por-Ron |
| {Fre, Ita, Ron} | Ron (4) | Fre-Ita closer (both Italo-Western) |

### Report format

The output of the experiment should be a single table and summary:

```
=== RESULTS: {model_name}, Layer {layer_label} ===

TOPOLOGY RANKING (by composition score, lower is better):

Rank | Score  | Splits                              | Matches ground truth?
-----+--------+-------------------------------------+---------------------
  1  | 0.0342 | {0,1}|{2,3,4}, {0,1,2}|{3,4}       | ✓ CORRECT
  2  | 0.0389 | {0,1}|{2,3,4}, {0,1,2,3}|{4}        | 
  3  | 0.0412 | ...                                  |
 ...
 15  | 0.0891 | ...                                  |

OUTGROUP PREDICTION ACCURACY (vote method):
Correct outgroup predictions: 8/10
Triples where prediction matched:
  {Spa,Por,Fre} → predicted Fre ✓
  {Spa,Por,Ita} → predicted Ita ✓
  ...

PAIRWISE RECONSTRUCTION ERROR (sanity check):
Spa-Por: 0.023  (expected: low)
Spa-Fre: 0.031
...
Spa-Ron: 0.048  (expected: high)
```

**Run this for all 4 layers × 2 models = 8 configurations.** Report all 8.

---

## Decision Criteria

| Outcome | Interpretation |
|---------|----------------|
| Correct topology is rank 1 in ≥1 configuration | Strong viability signal. Proceed. |
| Correct topology is rank 1–3 in most configurations | Viable. Signal exists but may need projection work. |
| Correct topology is rank 4–7 consistently | Ambiguous. May still work with refinements but not a clear signal. |
| Correct topology is rank 8+ everywhere | Not viable. The composition error framework does not carry phylogenetic information in these models. Stop. |

---

## Implementation Checklist

- [ ] Load FLORES-200, extract sentences for 5 Romance languages
- [ ] Load XLM-R-large in fp16, run inference, save representations at 4 layers
- [ ] Load Gemma-2-2B in fp16, run inference, save representations at 4 layers
- [ ] For each (model, layer): center data, fit 10 Procrustes maps on train split
- [ ] Compute 30 composition errors on test split (10 triples × 3 intermediates)
- [ ] Enumerate all 15 topologies, score each with both methods
- [ ] Output ranked topology table for all 8 configurations
- [ ] Verify ground truth topology's splits are `({0,1},{2,3,4})` and `({0,1,2},{3,4})`
- [ ] Run sanity checks (cosine similarity ordering, reconstruction error ordering)

---

## Common Pitfalls to Avoid

**Tokenization padding.** Gemma-2 pads on the right. Make sure the mean pool ignores padding tokens. Use the attention mask: `(hidden_states * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1, keepdim=True)`.

**CLS/BOS tokens.** For XLM-R, the `[CLS]` token (position 0) encodes a disproportionate language-identity signal. Exclude it from the mean pool. Slice from position 1 onward before pooling. Same for position 0 (`<bos>`) in Gemma-2.

**Numerical precision.** Fit Procrustes in float64. The representations come out of the model in float16, convert to float32 for storage, then cast to float64 before the SVD. Float32 SVD can be noisy for high-dimensional matrices.

**Representation scale.** Center the data (subtract mean) before fitting Procrustes. Orthogonal Procrustes finds rotations and reflections, not translations. If you don't center, the mean offset between languages dominates the fit and you're mostly fitting a translation, not a rotation.

**Direction consistency.** Define a convention: `R_AB` maps A→B. For a pair (A,B) where you only fit the forward direction, get the reverse as `R_BA = R_AB.T`. Do not fit both directions independently — independent fits on finite data won't be exact transposes and this introduces inconsistency that corrupts composition error measurements.

**Memory.** Don't hold all layers of all languages in GPU memory simultaneously. Extract representations layer by layer, move to CPU/numpy immediately. The linear algebra (SVD, matrix multiply) runs on CPU in numpy and is fast for these dimensions.
