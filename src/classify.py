"""Family discriminants on top of hub-aligned form residuals.

Two uses:

- **Tier 1 (held-out)**: train a 2-class discriminant (Germanic vs Romance)
  on form residuals of the 8 clean Germanic+Romance anchors, then classify
  English's residuals. English is never in the training data; the question
  is whether geometry overrides surface on an unseen language.
- **Tier 2b (universal axis)**: train the same Germanic-vs-Romance
  discriminant, then *transfer* it to other conflict cases (Maltese, Urdu,
  Romanian) — asking whether the SAME axis correctly orients languages in
  unrelated families.

We use LDA by default — it gives a single discriminant direction (the
"inherited-vs-borrowed axis" if it generalizes), which is the natural object
to project Tier 3's continuous placement onto. Logistic regression with L2
is offered as an alternative; the spec accepts either.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


@dataclass
class Discriminant:
    classifier: object
    classes_: np.ndarray
    feature_kind: str  # "form_residual" or "surface_tfidf" — for logging

    def predict(self, X) -> np.ndarray:
        return self.classifier.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.classifier.predict_proba(X)

    def axis(self) -> np.ndarray | None:
        """Single-direction axis if available (LDA), else None."""
        if hasattr(self.classifier, "scalings_"):
            return self.classifier.scalings_[:, 0]
        if hasattr(self.classifier, "coef_") and self.classifier.coef_.shape[0] == 1:
            return self.classifier.coef_[0]
        return None

    def project(self, X) -> np.ndarray:
        """Signed projection onto the discriminant axis (for Tier 3)."""
        ax = self.axis()
        if ax is None:
            raise ValueError("classifier has no single axis (multi-class logistic)")
        return X @ ax


def stack_residuals(
    residuals_by_lang: dict[str, np.ndarray],
    lang_to_label: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack per-language sentence residuals into (X, y, lang_row).

    Only includes languages present in ``lang_to_label``. ``lang_row[i]`` is
    the name of the source language for row i — used to do per-language
    metrics on the test side.
    """
    Xs, ys, lang_rows = [], [], []
    for L, label in lang_to_label.items():
        if L not in residuals_by_lang:
            continue
        R = residuals_by_lang[L]
        Xs.append(R)
        ys.append(np.full(R.shape[0], label, dtype=object))
        lang_rows.append(np.full(R.shape[0], L, dtype=object))
    return (np.concatenate(Xs, axis=0),
            np.concatenate(ys, axis=0),
            np.concatenate(lang_rows, axis=0))


def fit_family_discriminant(
    train_residuals_by_lang: dict[str, np.ndarray],
    lang_to_family: dict[str, str],
    method: str = "lda",
) -> Discriminant:
    """Fit a family discriminant on the *train* split of the supplied langs.

    ``train_residuals_by_lang`` must be train-split only — we don't slice
    here so the caller stays in control of the train/test boundary.
    """
    X, y, _ = stack_residuals(train_residuals_by_lang, lang_to_family)
    if method == "lda":
        clf = LinearDiscriminantAnalysis(solver="svd")
        clf.fit(X, y)
    elif method == "logreg":
        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=5000,
                                 solver="lbfgs")
        clf.fit(X, y)
    else:
        raise ValueError(f"unknown method {method!r}")
    return Discriminant(classifier=clf, classes_=clf.classes_,
                        feature_kind="form_residual")


def classify_language(
    discriminant: Discriminant,
    test_residuals_for_lang: np.ndarray,
) -> dict:
    """Return per-sentence predictions + summary stats for one held-out lang.

    Summary stats:
      - ``frac_by_class``: dict class -> fraction predicted that class
      - ``predictions``: (n_test,) array of predicted class labels
      - ``proba``: (n_test, n_classes) if available
    """
    preds = discriminant.predict(test_residuals_for_lang)
    classes = discriminant.classes_
    frac = {str(c): float(np.mean(preds == c)) for c in classes}
    out = {"frac_by_class": frac, "predictions": preds}
    try:
        out["proba"] = discriminant.predict_proba(test_residuals_for_lang)
    except (AttributeError, NotImplementedError):
        pass
    return out
