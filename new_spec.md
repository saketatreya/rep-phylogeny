# Genealogy-Over-Surface: Build Spec

## The result we are trying to show

A multilingual model trained only to predict the next token organizes languages by **descent**, not **appearance**. The test case: English is genealogically Germanic but its vocabulary is majority Romance (≈28% French + ≈28% Latin vs ≈25% Germanic in the dictionary). A surface-similarity measure therefore pulls English toward French. We test whether the model's representation geometry pulls it toward German instead — and whether the *same* geometric "inherited-vs-borrowed" axis simultaneously and correctly orients every other case where surface and genealogy conflict (Maltese → Semitic despite Italian vocabulary and Latin script; Romanian → Romance despite Slavic admixture; Urdu → Indo-Aryan/Hindi despite Perso-Arabic vocabulary and script).

There are three escalating claims, each its own gate:

- **Tier 0 (atomic flip):** For {English, German, French}, surface says English≈French, geometry says English≈German. One sign flip. Afternoon's work. If this fails, stop and diagnose pooling.
- **Tier 1 (held-out):** Build a Germanic-vs-Romance axis on *clean* languages (where descent and surface agree), never showing it English, then classify English. Geometry → Germanic; surface → Romance. Bootstrap for significance.
- **Tier 2 (universal axis — the prize):** A *single* inherited-vs-borrowed direction, found on Germanic/Romance, that also correctly orients Maltese, Romanian, and Urdu in their own families. If one axis transfers across unrelated families, the model didn't memorize family labels — it built a general, reusable representation of descent-vs-contact. That is the eureka.
- **Tier 3 (continuous — the dream, optional):** English isn't just on the Germanic side; its displacement along the axis tracks its actual historical borrowing fraction. Languages sit at the right *intermediate* points. Unobserved/ancestral states become geometrically locatable.

Framing note for continuity: the geometry distance below is the **pairwise base case of the composition method** — a single fitted transformation, where the residual measures how alignable two languages are. Act 1 (this spec) shows the strong result with the simplest version of the method. Act 2 (later) uses composition/residual structure for branching *order within* a family, which distance alone cannot give. Keep this language in the writeup.

---

## Reuse

The existing pipeline already has: FLORES tarball loader, representation extraction (mean-pool, last-token), orthogonal Procrustes fitting (centered, float64 SVD), per-sentence bootstrap. Reuse all of it. This spec adds: a larger language set, a surface n-gram distance, hub alignment, a held-out family discriminant, the single-axis test, NJ tree construction, reference-tree comparison, and transliteration controls.

Same numerical discipline as before: center before SVD, float64 for all linear algebra, save every intermediate (.npy for representations, .npz for matrices, .csv for tables, .txt logs). We lost data once; save everything to `/kaggle/working/results/` and download.

---

## Language set

FLORES-200 devtest, 1012 parallel sentences. Model: **XLM-R-large**, fp16 (the clean instrument from prior runs; Gemma-2 is excluded — its Romanian-intermediate bias is a constant offset that contaminates exactly the conflict cases).

**Conflict cases (the stars — surface and genealogy disagree):**

| Language | FLORES code | Genealogy | Surface pull | Why it's a conflict |
|----------|-------------|-----------|--------------|---------------------|
| English | `eng_Latn` | West Germanic | Romance | ~56% Romance vocabulary, Latin script |
| Maltese | `mlt_Latn` | Semitic | Romance | Heavy Italian/Sicilian vocabulary, Latin script (script works *against* correct answer) |
| Romanian | `ron_Latn` | Romance | mixed/Slavic | Slavic admixture in vocabulary |
| Urdu | `urd_Arab` | Indo-Aryan (≈Hindi) | Persian/Arabic | Perso-Arabic vocabulary + script (script works against) |

**Clean anchors (descent and surface agree — used to build axes):**

| Family | Languages (FLORES codes) |
|--------|--------------------------|
| Germanic | German `deu_Latn`, Dutch `nld_Latn`, Swedish `swe_Latn`, Danish `dan_Latn` |
| Romance | French `fra_Latn`, Spanish `spa_Latn`, Italian `ita_Latn`, Portuguese `por_Latn` |
| Slavic | Russian `rus_Cyrl`, Polish `pol_Latn`, Czech `ces_Latn`, Bulgarian `bul_Cyrl` |
| Semitic | Arabic `arb_Arab`, Hebrew `heb_Hebr` |
| Indo-Aryan | Hindi `hin_Deva`, Bengali `ben_Beng`, Nepali `npi_Deva` |
| Iranian | Persian `pes_Arab` |
| Uralic control (non-IE) | Finnish `fin_Latn`, Hungarian `hun_Latn`, Estonian `est_Latn` |

**Verify every code against the actual devtest file list before extraction** — FLORES codes have script suffixes and a few are non-obvious. Log the resolved file path for each language. The Uralic control exists to confirm the axis isn't just "Indo-European-ness"; Finnish/Hungarian/Estonian should sit clearly apart from everything IE.

Hindi and Urdu are the jewel: essentially one spoken language in two scripts with two prestige vocabularies. The dictionary-and-script view rips them across the Indo-Aryan/Iranian divide; a linguist collapses them. If geometry pulls them together *despite different scripts*, that is the single most striking data point. (Hindi is a clean anchor for Indo-Aryan; Urdu is the conflict case tested against it.)

---

## The two distances

Everything below is computed per (layer, pooling) configuration. Sweep layers and pooling (see "Sweeps"). All distances are computed on the **test split** (sentences 812–1011); any fitting (Procrustes, axis, discriminant) uses the **train split** (0–811).

### Surface distance (the baseline that should be fooled)

Character n-gram overlap on the **raw FLORES text** (not representations).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Per language, concatenate its 1012 raw sentences? No — keep per-sentence,
# then aggregate, so the measure is comparable to geometry which is per-sentence.
# Simplest robust version: build one char-ngram TF-IDF profile per language
# over its full devtest text, then cosine between language profiles.

def surface_distance_matrix(raw_text_by_lang, ngram_range=(3,5)):
    langs = list(raw_text_by_lang.keys())
    # One document per language = its concatenated devtest sentences
    docs = [" ".join(raw_text_by_lang[L]) for L in langs]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, min_df=2)
    X = vec.fit_transform(docs)            # (n_lang, n_features)
    sim = cosine_similarity(X)             # (n_lang, n_lang)
    dist = 1.0 - sim
    return langs, dist
```

This captures shared orthography + shared/borrowed vocabulary. For within-script conflicts (English, Maltese, Romanian) it is a genuine competitor. For cross-script pairs it saturates to ≈max distance (no shared characters) — expected, and handled by the transliteration control below.

**Precondition check (do this first, it gates the whole experiment):** confirm `surface_dist(Eng, Fra) < surface_dist(Eng, Deu)`, i.e. surface really does make English look more Romance than Germanic. If surface does *not* put English closer to French, the premise is wrong and we need to rethink before touching geometry. Log this explicitly.

### Geometry distance (the method's pairwise base case)

Two variants — compute **both**, we do not know a priori which carries the signal (lesson from prior runs: don't pre-judge).

**(a) Procrustes residual distance** — "how alignable are these two clouds (shape)":

```python
def procrustes_residual_distance(X_A, X_B, n_train=812):
    # X_A, X_B: (1012, d) parallel sentence reps, float64
    XA_tr, XA_te = X_A[:n_train], X_A[n_train:]
    XB_tr, XB_te = X_B[:n_train], X_B[n_train:]
    mA, mB = XA_tr.mean(0), XB_tr.mean(0)
    M = (XA_tr - mA).T @ (XB_tr - mB)
    U, _, Vt = svd(M, full_matrices=False)
    R = U @ Vt
    resid = np.linalg.norm((XA_te - mA) @ R - (XB_te - mB), axis=1)
    return resid.mean()          # lower = more alignable = closer
```

Note this is a *within-language-pair* comparison when used for the flip (English appears in both English-German and English-French), so English's own representation quality largely cancels — a desirable property that defuses the per-language-quality confound that bit us before.

**(b) Hub-centroid (form) distance** — "after a common alignment, how far apart are the language-specific offsets (location)":

Hub alignment = generalized **orthogonal** Procrustes (NOT affine — affine can rescale away the very low-variance direction that may carry the signal, the same way ridge erased it in prior runs). Single global scale normalization per language is allowed (removes gross magnitude, e.g. norm growth, without selectively killing a direction).

```python
def hub_align(reps_by_lang, n_train=812, n_iters=5):
    # reps_by_lang: dict lang -> (1012, d) float64
    langs = list(reps_by_lang.keys())
    # global scale normalize each language (one scalar)
    Xs = {}
    for L in langs:
        X = reps_by_lang[L].copy()
        s = np.sqrt((np.linalg.norm(X[:n_train] - X[:n_train].mean(0), axis=1)**2).mean())
        Xs[L] = X / s
    # initialize reference = mean over languages of train clouds (parallel => aligned by sentence)
    ref = np.mean([Xs[L][:n_train] for L in langs], axis=0)
    V = {L: np.eye(Xs[L].shape[1]) for L in langs}
    for _ in range(n_iters):
        for L in langs:
            M = (Xs[L][:n_train] - Xs[L][:n_train].mean(0)).T @ (ref - ref.mean(0))
            U, _, Vt = svd(M, full_matrices=False)
            V[L] = U @ Vt
        aligned_tr = [ (Xs[L][:n_train]) @ V[L] for L in langs ]
        ref = np.mean(aligned_tr, axis=0)
    # apply rotations to full data; do NOT per-language center (we need centroids)
    grand_mean = np.mean([ (Xs[L] @ V[L]) for L in langs ], axis=0).mean(0)
    aligned = {L: (Xs[L] @ V[L]) for L in langs}
    return aligned, grand_mean

def hub_centroid_distance_matrix(aligned, grand_mean, n_train=812):
    # form_L = centroid of language L on TEST split, relative to grand mean
    langs = list(aligned.keys())
    form = {L: aligned[L][n_train:].mean(0) - grand_mean for L in langs}
    n = len(langs); D = np.zeros((n, n))
    for i, A in enumerate(langs):
        for j, B in enumerate(langs):
            D[i, j] = np.linalg.norm(form[A] - form[B])
    return langs, D
```

Critical detail: for the hub-centroid distance, **do not center per language** — the language centroid relative to the grand mean *is* the genealogical signal. Center only by the grand mean. (For the Procrustes-residual distance, per-pair centering is correct and standard.)

---

## Pooling variants

Lead with plain **mean-pool** (excluding special tokens) so the result can't be called engineered. Also compute a **high-frequency-token pooling** variant, because genealogy lives in the borrowing-resistant grammatical core (function words, morphology) per Rabinovich et al. 2017 — and high-frequency subwords are a language-agnostic proxy for exactly those:

```python
# For each language, find its top-K most frequent subword token ids across devtest.
# Pool the layer's hidden states only over positions whose token id is in that set.
# K ~ 50–100. This concentrates on function words / inflectional morphology.
```

If plain mean-pool already gives the flip, high-frequency pooling is a *mechanistic confirmation* (signal strengthens when you restrict to conservative tokens) and becomes a result in its own right. If plain mean-pool is muddy (English syntax has drifted Romance-ward, so the sentence-level form signal may be faint), high-frequency pooling is the rescue. Report both either way.

---

## Tier 0 — Atomic flip {English, German, French}

The whole thing in one table. For each (layer, pooling):

```
                 surface_dist   procrustes_resid   hub_centroid
English–French   <should be LO>  ...                ...
English–German   <should be HI>  ...                ...
FLIP?            premise: E–F<E–G   eureka: E–F>E–G    eureka: E–F>E–G
```

- **Precondition (premise):** `surface(E,F) < surface(E,G)` — surface makes English look Romance.
- **Eureka condition:** `geometry(E,F) > geometry(E,G)` — geometry makes English look Germanic — for at least one geometry variant.
- **Gate:** if the flip appears at any (layer, pooling), proceed to Tier 1. If no flip anywhere, run the high-frequency-pooling variant and the layer sweep before concluding; if still nothing, the sentence-level form signal is too weak and we stop and reassess (this is itself an informative negative).

Report the flip at every layer/pooling so we see where it's cleanest, not just whether it exists.

---

## Tier 1 — Held-out family discriminant (8 clean + English held out)

Languages: Germanic {deu, nld, swe, dan} + Romance {fra, spa, ita, por}. English is **held out** — never used to define the axis.

Operate on **form residuals** to isolate language-specific structure from shared sentence meaning. After hub alignment, for each sentence index `i`, `meaning_i = mean over languages of aligned X_{L,i}`; `form_{L,i} = aligned X_{L,i} − meaning_i`. The form residual is what carries genealogy (the meaning is shared across the parallel translations and is genealogically inert).

```python
# Train a linear classifier (logistic regression or LDA) on form residuals of the
# 8 clean languages, label = family (Germanic=0, Romance=1), TRAIN split only.
# Then classify English's form residuals on the TEST split.
# Report fraction of English test sentences classified Germanic.
#
# Do the identical thing in SURFACE space: represent each sentence by its char-ngram
# vector, train family classifier on the 8 clean langs, classify English sentences.
```

Bootstrap: resample the test sentences 1000×, recompute the classified fraction each time.

- **Output:** `P(English → Germanic | geometry)` with bootstrap CI, vs `P(English → Romance | surface)` with bootstrap CI.
- **Gate:** geometry classifies English as Germanic in a large majority of resamples (target >90%) while surface classifies it Romance. The held-out construction is what makes this airtight — an axis built without English, on clean cases only, that then *overrides* English's surface similarity, cannot be dismissed as curve-fitting.

---

## Tier 2 — Universal inherited-vs-borrowed axis (the prize)

### 2a. Full distance matrices and NJ trees

Compute surface, Procrustes-residual, and hub-centroid distance matrices over all ~25 languages. Build a neighbor-joining tree from each (use `scipy`/`biotite`/`skbio` NJ, or `Bio.Phylo`).

**Headline figure:** the geometry NJ tree and the surface NJ tree side by side, leaves colored by true family. In the geometry tree, English sits inside Germanic, Maltese inside Semitic, Romanian inside Romance, Urdu beside Hindi. In the surface tree, English sits by French, Maltese by Italian, Urdu away from Hindi. The two trees disagree *exactly* on the conflict languages.

**Headline number:** bootstrap clade support. Resample test sentences 1000×, rebuild the geometry NJ tree, and report, for each conflict language, the fraction of resamples in which it falls within its **true** family clade — under geometry vs under surface.

```
                geometry P(true family)   surface P(true family)
English (Germanic)        0.9x                    0.0x
Maltese (Semitic)         0.9x                    0.0x
Romanian (Romance)        0.9x                    0.?x
Urdu (Indo-Aryan)         0.9x                    0.0x  (cross-script)
```

### 2b. The single shared axis (this is the eureka, if it holds)

Find ONE direction and show it orients every family's conflict case correctly.

```python
# Define the inherited-vs-borrowed axis using ONLY clean Indo-European anchors,
# as the direction separating each family's centroid from its sisters in hub space.
# Concretely: for each clean language compute its form centroid; fit the axis as the
# leading direction that aligns "distance to own-family centroid" small vs
# "distance to surface-nearest other-family centroid" — or more simply:
#
#   axis = direction along which a language's genealogical family is predicted,
#          learned by multinomial LDA on clean-language family labels in hub space.
#
# Then project EVERY conflict case onto this axis / classify with this discriminant:
#   English   -> Germanic side?
#   Maltese   -> Semitic side?     (axis built on IE only — does it transfer?)
#   Urdu      -> Indo-Aryan side?
#   Romanian  -> Romance side?
```

The strong version: build the discriminant on Germanic-vs-Romance **only**, then test whether the *same* geometry that separates Germanic from Romance also places Maltese on the Semitic side of its relevant contrast and Urdu with Hindi. If a discriminant trained on one family contrast transfers to correctly orient conflict cases in *unrelated* families, the model has a general, reusable "descent vs contact" representation — not memorized family labels. Report transfer accuracy per conflict case, with bootstrap CIs, against the surface baseline doing the same transfer (which should fail).

Be honest about the risk: the axis may be partly family-specific and not transfer cleanly. If so, the per-family flips (each family's own axis correctly orienting its own conflict case) are still a strong result — just report it as "consistent per-family" rather than "single universal axis." Both are publishable; only the universal version is the capital-E eureka.

### 2c. Reference-tree comparison

- **Glottolog** gold tree → categorical clade membership (used in 2a clade support).
- **ASJP** database → continuous genealogical distances from core (Swadesh-list) vocabulary, i.e. the linguist's borrowing-resistant distance. Mantel-test each distance matrix against ASJP: geometry should correlate strongly; surface should correlate well on clean languages but *break down on the conflict subset*. Report Mantel `r` overall and on the conflict-language submatrix separately — the conflict submatrix is where geometry should beat surface decisively.

### 2d. Transliteration control (cross-script conflicts)

For Urdu/Hindi and Persian/Arabic, script could let the model get the right answer for the wrong reason (or, for surface, the wrong answer trivially). Romanize all relevant languages (e.g. `uroman` or ISO translit) and recompute **surface** distance on the romanized text. This makes Hindi and Urdu surface-similar. Then show geometry still groups Hindi+Urdu tighter than even the romanized-surface measure does, and that geometry's grouping was not driven by script (recompute geometry on a script-controlled extraction if feasible, or argue from the romanized-surface contrast). Note for the writeup: script works *against* the correct answer for Maltese (Latin, like Romance) and Urdu (Arabic, like Persian), so geometry succeeding there is *stronger* evidence — but you must run this control to prove it isn't a script artifact.

---

## Tier 3 — Continuous placement (optional, the dream)

Quantify each conflict language's position *along* the inherited-vs-borrowed axis (signed projection), not just which side. Then test whether displacement tracks the actual historical borrowing fraction.

- Borrowing fractions from the **World Loanword Database (WOLD)** (Haspelmath & Tadmor) where available, or etymological vocabulary breakdowns.
- Prediction: English sits Germanic but displaced toward Romance further than Dutch/German; the displacement is monotone in borrowing fraction across languages. If geometry places languages at the correct *intermediate* points, the axis is a genuine continuous parameterization — and the extrapolated inherited pole is where a proto-language would sit. This is the "geometrically locatable unobserved states" claim from the proposal, made literal. Report Spearman correlation between axis displacement and WOLD borrowing rate.

---

## Sweeps

- **Layers:** {4, 8, 12, 18} for XLM-R-large. Genealogical *form* signal is expected stronger at shallow-to-mid layers (before meaning abstraction dominates) but too-shallow collapses to pure script/lexical surface. Report the flip strength and Tier-1 held-out accuracy per layer; pick the cleanest for the headline, report all.
- **Pooling:** {mean-pool, high-frequency-token}. As above.
- **Geometry variant:** {Procrustes residual, hub centroid}. As above.

Do not tune these to maximize the result and then report only the winner — report the full sweep and identify the headline configuration honestly, with the others shown.

---

## Outputs

```
results/
  surface/                 distance matrices (.npz), precondition check (.txt)
  geometry/                per (layer,pool,variant): distance matrices (.npz)
  hub/                     aligned reps, V rotations, form residuals
  tier0_atomic.csv         the flip table, all layers/pools
  tier1_heldout.csv        P(Eng->Germanic) geometry vs surface, bootstrap CIs
  tier2_clade_support.csv  per conflict language, geometry vs surface, bootstrap
  tier2_axis_transfer.csv  single-axis transfer accuracy per conflict case
  tier2_mantel.csv         Mantel r vs ASJP, overall and conflict submatrix
  trees/                   NJ trees (Newick) for surface + each geometry variant
  figures/                 side-by-side NJ trees, axis-projection plots
  SUMMARY.txt              the decision table below
  *.log                    full per-stage logs
```

`SUMMARY.txt` decision table, read top to bottom:

```
1. PRECONDITION: surface(Eng,Fra) < surface(Eng,Deu)?   [Y/N]   <- if N, premise dead
2. TIER 0 FLIP: geometry(Eng,Fra) > geometry(Eng,Deu)?  [Y/N, best config]
3. TIER 1 HELD-OUT: P(Eng->Germanic|geom) vs P(Eng->Romance|surf), CIs
4. TIER 2 CLADE SUPPORT: per conflict lang, geom vs surf
5. TIER 2 UNIVERSAL AXIS: does IE-trained axis orient Maltese/Urdu correctly?  <- the prize
6. TIER 2 MANTEL: geom vs surf correlation to ASJP, conflict submatrix
7. CONTROLS: transliteration (script not driving it), Uralic apart from IE
```

---

## What each tier buys, and the honest failure modes

- Tier 0 flip present → the effect exists in elemental form. Absent everywhere → sentence-level form signal too weak; report as negative and reconsider (function-word extraction, token-level rather than sentence-level).
- Tier 1 held-out positive → not curve-fitting; the geometry genuinely overrides surface on an unseen language. This alone is a strong, defensible result and the minimum publishable finding.
- Tier 2 per-family flips positive but axis doesn't transfer → "geometry tracks genealogy over surface across multiple families," strong, publishable, not the universal-axis eureka.
- Tier 2 universal axis transfers across families → the eureka: a single, reusable, descent-vs-contact geometric primitive learned from next-token prediction. This is the result that motivates the whole direction and earns Act 2 (composition for within-family branching order).
- Tier 3 monotone in borrowing fraction → continuous parameterization; ancestral states locatable. The dream; not required for viability.

Lead the paper and the demo with the English flip — there is no similarity story for why English should sit with German rather than French, so a skeptic has nothing to hide behind. Build toward the universal axis.
