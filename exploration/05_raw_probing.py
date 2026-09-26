"""
05_raw_probing.py

Counterpart to 05_probing.py -- probing the SHARED subspace and raw
activations for cross-factor decodability.

WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW
WHY WE DO THIS
WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW

05_probing.py has a structural blind spot: its cross-probe trains on AST-pure
dims to predict builtin (and vice versa), but those dims are DEFINED to be
uninformative about the other factor by the VP purity criterion.  Near-chance
cross-probe results are therefore a tautology of the mask definition, not
evidence of independence.

This script performs three analyses on unmasked or shared-space activations:

  ANALYSIS 1 -- SHARED SUBSPACE PROBE
  ------------------------------------
  Train linear probes on the SHARED subspace dims (dims jointly explained
  by both AST and builtin, as flagged by step 02 VP) to predict (a) AST
  node and (b) builtin.

  If shared dims are a true interaction circuit: both AST and builtin should
  be decodable from them simultaneously.

  If shared dims are dominated by one factor: only that factor decodes well.

  ANALYSIS 2 -- RAW CROSS-PROBE
  ------------------------------
  Train on RAW full-dimensional activations (no masking) to predict each
  factor.  Key comparison:

    raw dims -> predict AST     (expected: high -- AST is well-encoded)
    raw dims -> predict builtin (key question: is builtin encoded at all
                                 in raw space, even weakly?)

  Unlike the pure-dims cross-probe in 05_probing.py, this includes all
  signal (pure + shared + noise), giving the correct upper bound on
  cross-factor leakage.

  ANALYSIS 3 -- SHARED SUBSPACE AUROC
  ------------------------------------
  Concept-level AUROC per AST node and per builtin, computed on shared
  subspace activations only.  Compare with 05_probing.py AUROC (raw dims)
  to see how much of the decodability lives in the shared circuit.

WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW
HOW IT IS COMPUTED
WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW

Linear probe:
  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
  StratifiedKFold(n_splits=5) -- mean CV accuracy reported.
  Falls back to chance (1/n_classes) if too few samples per class.

AUROC:
  Concept direction = mean residual vector over baseline stubs (unit-normalised
  per layer).  Cosine similarity of every prompt to that direction.
  sklearn.metrics.roc_auc_score (binary: positive = prompts with this concept).

WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW
INPUT FILES  (from steps 01 and 02)
WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW

  01_<stem>_residual_all.npy     float32  (N, L+1, H)
  01_<stem>_meta.json            prompt metadata
  02_<stem>_purity_masks.npz     boolean purity masks (required for shared dims)

WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW
OUTPUT FILES  (prefix 05_raw_<stem>_)
WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW WWW

  05_raw_<stem>_shared_probe_accuracy.npz
      Keys: shared_ast, shared_builtin -- shape (L+1,) CV accuracy on
      shared subspace dims for each target.

  05_raw_<stem>_shared_probe_folds.npz
      Keys: shared_ast, shared_builtin -- shape (L+1, n_splits) per-fold
      accuracy for downstream ANOVA input.

  05_raw_<stem>_raw_crossprobe.json
      Per-layer: raw dims predicting AST and builtin, chance levels,
      and leakage above chance for each target.

  05_raw_<stem>_shared_auroc_ast.npz
      Per-AST-node AUROC computed on shared subspace activations.
      One key per concept, shape (L+1,).

  05_raw_<stem>_shared_auroc_builtin.npz
      Per-builtin AUROC on shared subspace. One key per builtin, shape (L+1,).

  05_raw_<stem>_shared_vs_pure_comparison.json
      Per-layer comparison: raw / shared accuracy for both targets.
      Optionally includes pure-subspace results from 05_probing.py.

Usage
-----
  python 05_raw_probing.py --stem contrastive_stubs
  python 05_raw_probing.py --stem contrastive_stubs --all_concepts --plot
"""

from __future__ import annotations

import os
import argparse
import json
import warnings
from pathlib import Path

# Force single-threaded numeric backend in shared compute environments.
for _k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENMP_NUM_THREADS"]:
    os.environ.setdefault(_k, "1")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score


# -----------------------------------------------------------------------------
# Probe utilities
# -----------------------------------------------------------------------------

def _train_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    C: float = 1.0,
    seed: int = 42,
) -> tuple[float, np.ndarray]:
    """
    Stratified k-fold logistic regression probe.
    Returns (mean_accuracy, per_fold_accuracies).
    Falls back to chance if too few samples per class.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2 or counts.min() < n_splits:
        chance = 1.0 / max(len(classes), 1)
        return chance, np.full(n_splits, chance)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for tr, te in skf.split(X, y):
            try:
                clf = LogisticRegression(
                    C=C, max_iter=1000, solver="lbfgs",
                    multi_class="multinomial", n_jobs=1,
                )
            except TypeError:
                clf = LogisticRegression(
                    C=C, max_iter=1000, solver="lbfgs", n_jobs=1,
                )
            clf.fit(X[tr], y[tr])
            scores.append(accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(scores)), np.array(scores)


def _select_dims(act: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Select columns of act where mask is True. Return act unchanged if mask empty."""
    if mask.sum() == 0:
        return act
    return act[:, mask]


# -----------------------------------------------------------------------------
# Load utilities
# -----------------------------------------------------------------------------

def _load_masks(data_dir: Path, stem: str) -> dict | None:
    path = data_dir / f"02_{stem}_purity_masks.npz"
    if not path.exists():
        print(f"  WARNING: purity masks not found at {path} -- shared probing skipped")
        return None
    npz = np.load(path)
    out = {}
    for key in ("ast_pure", "builtin_pure", "shared", "interaction", "noise"):
        fkey = f"residual_{key}"
        if fkey in npz.files:
            out[key] = npz[fkey].astype(bool)   # (L+1, H)
    return out or None


def _get_explicit_labels(meta: list[dict]) -> tuple[list, list, list]:
    """
    Returns (ast_labels, builtin_labels, explicit_idxs) for non-pseudo
    explicit stubs only.
    """
    idxs, ast_labs, b_labs = [], [], []
    for i, m in enumerate(meta):
        if m.get("variant_type", "explicit") == "explicit":
            ast_n = m["ast_node"]
            b_n   = m["builtin_obj"]
            if ast_n.startswith("__") or b_n.startswith("__"):
                continue
            idxs.append(i)
            ast_labs.append(ast_n)
            b_labs.append(b_n)
    return ast_labs, b_labs, idxs


def _shared_mask_at(masks: dict, layer: int, H: int) -> np.ndarray:
    """Union of shared + interaction dims at one layer."""
    m = np.zeros(H, bool)
    if "shared" in masks:
        m |= masks["shared"][layer]
    if "interaction" in masks:
        m |= masks["interaction"][layer]
    return m


# -----------------------------------------------------------------------------
# Analysis 1: Shared subspace probe
# -----------------------------------------------------------------------------

def run_shared_probe(
    resid_all: np.ndarray,
    ast_labels: list,
    builtin_labels: list,
    explicit_idxs: list[int],
    masks: dict,
    n_splits: int = 5,
) -> dict:
    """
    Per-layer logistic regression probe on shared subspace dims.
    Returns accuracy and fold scores for AST and builtin targets.
    """
    N, L1, H = resid_all.shape
    le_ast = LabelEncoder().fit(ast_labels)
    le_b   = LabelEncoder().fit(builtin_labels)
    y_ast  = le_ast.transform(ast_labels)
    y_b    = le_b.transform(builtin_labels)

    acc_ast  = np.zeros(L1);  fold_ast = np.zeros((L1, n_splits))
    acc_b    = np.zeros(L1);  fold_b   = np.zeros((L1, n_splits))
    n_dims   = np.zeros(L1, int)

    act_sub = resid_all[explicit_idxs]   # (n_explicit, L+1, H)

    for l in range(L1):
        print(f"  Shared probe layer {l+1}/{L1}...", end="\r")
        m = _shared_mask_at(masks, l, H)
        n_dims[l] = int(m.sum())
        X = _select_dims(act_sub[:, l, :], m)

        a, af = _train_probe(X, y_ast, n_splits)
        b, bf = _train_probe(X, y_b,   n_splits)
        acc_ast[l] = a;  fold_ast[l] = af
        acc_b[l]   = b;  fold_b[l]   = bf
    print()

    return {
        "accuracy":    {"shared_ast": acc_ast,  "shared_builtin": acc_b},
        "fold_scores": {"shared_ast": fold_ast, "shared_builtin": fold_b},
        "n_shared_dims":  n_dims,
        "chance_ast":     1.0 / len(le_ast.classes_),
        "chance_builtin": 1.0 / len(le_b.classes_),
    }


# -----------------------------------------------------------------------------
# Analysis 2: Raw cross-probe
# -----------------------------------------------------------------------------

def run_raw_crossprobe(
    resid_all: np.ndarray,
    ast_labels: list,
    builtin_labels: list,
    explicit_idxs: list[int],
    n_splits: int = 5,
) -> list[dict]:
    """
    Per-layer probe on raw (full-dimensional) activations predicting both
    AST node and builtin.  Reports accuracy and leakage above chance.
    """
    N, L1, H = resid_all.shape
    le_ast = LabelEncoder().fit(ast_labels)
    le_b   = LabelEncoder().fit(builtin_labels)
    y_ast  = le_ast.transform(ast_labels)
    y_b    = le_b.transform(builtin_labels)
    chance_ast = 1.0 / len(le_ast.classes_)
    chance_b   = 1.0 / len(le_b.classes_)

    act_sub = resid_all[explicit_idxs]
    results = []
    for l in range(L1):
        print(f"  Raw cross-probe layer {l+1}/{L1}...", end="\r")
        X = act_sub[:, l, :]
        acc_ast, _ = _train_probe(X, y_ast, n_splits)
        acc_b,   _ = _train_probe(X, y_b,   n_splits)
        results.append({
            "layer":               l,
            "raw_predict_ast":     float(acc_ast),
            "raw_predict_builtin": float(acc_b),
            "chance_ast":          chance_ast,
            "chance_builtin":      chance_b,
            "leak_raw_to_ast":     float(acc_ast - chance_ast),
            "leak_raw_to_builtin": float(acc_b   - chance_b),
        })
    print()
    return results


# -----------------------------------------------------------------------------
# Analysis 3: Shared subspace AUROC
# -----------------------------------------------------------------------------

def _concept_dir(
    resid_all: np.ndarray,
    meta: list[dict],
    concept_name: str,
    concept_type: str,   # "ast" | "builtin"
) -> np.ndarray | None:
    """Unit-normalised mean activation from baseline stubs.  Returns (L+1, H)."""
    vtype     = "baseline_ast" if concept_type == "ast" else "baseline_builtin"
    label_key = "ast_node"     if concept_type == "ast" else "builtin_obj"
    idxs = [i for i, m in enumerate(meta)
            if m.get("variant_type") == vtype and m.get(label_key) == concept_name]
    if not idxs:
        return None
    mean_vec = resid_all[idxs].mean(axis=0)
    norms    = np.linalg.norm(mean_vec, axis=-1, keepdims=True).clip(1e-12)
    return (mean_vec / norms).astype(np.float32)


def _auroc_shared(
    resid_all:   np.ndarray,
    meta:        list[dict],
    concept_name: str,
    concept_type: str,
    concept_dir:  np.ndarray,   # (L+1, H)
    mask_all:     np.ndarray,   # (L+1, H) boolean  -- shared dims per layer
) -> np.ndarray:
    """
    AUROC per layer restricted to shared subspace activations.
    Positive = prompts with this concept; negative = all others.
    """
    label_key = "ast_node" if concept_type == "ast" else "builtin_obj"
    pos_idx = [i for i, m in enumerate(meta) if m.get(label_key) == concept_name]
    neg_idx = [i for i, m in enumerate(meta) if m.get(label_key) != concept_name]
    if not pos_idx:
        return np.full(resid_all.shape[1], 0.5, dtype=np.float32)

    all_idx = pos_idx + neg_idx
    y_true  = np.array([1] * len(pos_idx) + [0] * len(neg_idx))
    L1      = resid_all.shape[1]
    auroc   = np.zeros(L1, dtype=np.float32)

    for l in range(L1):
        m = mask_all[l]
        if m.sum() == 0:
            auroc[l] = 0.5
            continue
        act_l  = resid_all[all_idx, l, :][:, m]
        dir_l  = concept_dir[l, m]
        norms  = np.linalg.norm(act_l, axis=1).clip(1e-12)
        dir_n  = dir_l / (np.linalg.norm(dir_l) + 1e-12)
        scores = (act_l / norms[:, None]) @ dir_n
        try:
            auroc[l] = float(roc_auc_score(y_true, scores))
        except Exception:
            auroc[l] = 0.5
    return auroc


def run_shared_auroc(
    resid_all: np.ndarray,
    meta: list[dict],
    masks: dict,
    concept_type: str,
) -> dict[str, np.ndarray]:
    """AUROC in shared subspace for all concepts of concept_type."""
    label_key = "ast_node" if concept_type == "ast" else "builtin_obj"
    concepts  = sorted({m[label_key] for m in meta
                        if not m.get(label_key, "").startswith("__")})
    N, L1, H  = resid_all.shape

    # Build combined shared+interaction mask (L+1, H)
    mask_all = np.zeros((L1, H), bool)
    if "shared" in masks:
        mask_all |= masks["shared"]
    if "interaction" in masks:
        mask_all |= masks["interaction"]

    out = {}
    for c in concepts:
        cdir = _concept_dir(resid_all, meta, c, concept_type)
        if cdir is None:
            continue
        auroc = _auroc_shared(resid_all, meta, c, concept_type, cdir, mask_all)
        out[c] = auroc
        print(f"  {concept_type.upper()} '{c}': AUROC_final={auroc[-1]:.3f}", end="\r")
    print()
    return out


# -----------------------------------------------------------------------------
# Comparison table
# -----------------------------------------------------------------------------

def build_comparison(
    shared_probe: dict | None,
    raw_crossprobe: list[dict],
    pure_probe_path: Path | None,
) -> list[dict]:
    """Per-layer table: raw / shared accuracy. Optionally includes pure-subspace."""
    L1 = len(raw_crossprobe)
    pure_ast = pure_b = None
    if pure_probe_path and pure_probe_path.exists():
        npz = np.load(pure_probe_path)
        if "ast_pure_ast" in npz.files:
            pure_ast = npz["ast_pure_ast"]
        if "builtin_pure_builtin" in npz.files:
            pure_b = npz["builtin_pure_builtin"]

    rows = []
    for l in range(L1):
        row = {
            "layer":          l,
            "raw_ast":        float(raw_crossprobe[l]["raw_predict_ast"]),
            "raw_builtin":    float(raw_crossprobe[l]["raw_predict_builtin"]),
            "chance_ast":     float(raw_crossprobe[l]["chance_ast"]),
            "chance_builtin": float(raw_crossprobe[l]["chance_builtin"]),
        }
        if shared_probe:
            row["shared_ast"]     = float(shared_probe["accuracy"]["shared_ast"][l])
            row["shared_builtin"] = float(shared_probe["accuracy"]["shared_builtin"][l])
        if pure_ast is not None:
            row["pure_ast_predict_ast"] = float(pure_ast[l])
        if pure_b is not None:
            row["pure_builtin_predict_builtin"] = float(pure_b[l])
        rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

COLOURS = {
    "raw":    "#607D8B",
    "shared": "#6A1B9A",
    "pure":   "#1565C0",
    "chance": "#BDBDBD",
}


def plot_probe_comparison(comparison: list[dict], save_path: Path | None = None) -> None:
    layers = [r["layer"] for r in comparison]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, target, label in [
        (axes[0], "ast",     "Predict AST node"),
        (axes[1], "builtin", "Predict builtin"),
    ]:
        ax.plot(layers, [r[f"raw_{target}"] for r in comparison],
                "o-", lw=2, color=COLOURS["raw"], label="raw")
        if "shared_ast" in comparison[0]:
            ax.plot(layers, [r[f"shared_{target}"] for r in comparison],
                    "o-", lw=2, color=COLOURS["shared"], label="shared subspace")
        key_pure = "pure_ast_predict_ast" if target == "ast" else "pure_builtin_predict_builtin"
        if key_pure in comparison[0]:
            ax.plot(layers, [r[key_pure] for r in comparison],
                    "o--", lw=1.5, color=COLOURS["pure"], label="pure subspace")
        ax.axhline(comparison[0][f"chance_{target}"], color=COLOURS["chance"],
                   lw=1, linestyle=":", label="chance")
        ax.set_xlabel("Layer")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel("CV accuracy")
    fig.suptitle("Probe accuracy: raw vs shared subspace", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_auroc_comparison(
    raw_auroc: dict,
    shared_auroc: dict,
    concept_type: str,
    top_k: int = 10,
    save_path: Path | None = None,
) -> None:
    """Top-k concepts by final-layer raw AUROC: raw vs shared AUROC per layer."""
    if not raw_auroc or not shared_auroc:
        return
    common = sorted(set(raw_auroc) & set(shared_auroc),
                    key=lambda c: -float(raw_auroc[c][-1]))[:top_k]
    if not common:
        return
    L1     = len(next(iter(raw_auroc.values())))
    layers = np.arange(L1)
    cmap   = plt.colormaps["tab10"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for i, c in enumerate(common):
        col = cmap(i % 10)
        axes[0].plot(layers, raw_auroc[c],    "-",  color=col, lw=1.5, label=c)
        axes[1].plot(layers, shared_auroc[c], "--", color=col, lw=1.5, label=c)

    for ax, title in [(axes[0], f"Raw ({concept_type})"),
                      (axes[1], f"Shared subspace ({concept_type})")]:
        ax.axhline(0.5, color="#BDBDBD", lw=0.8, linestyle=":")
        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.4, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.2)

    fig.suptitle(f"AUROC: raw vs shared -- {concept_type} (top {top_k})", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Raw probing counterpart to 05_probing.py")
    parser.add_argument("--stem",         default="contrastive_stubs")
    parser.add_argument("--in_dir", "-d", default="data",
                        help="Directory containing 01_* and 02_* input files "
                             "(relative to script dir, default: data)")
    parser.add_argument("--out_dir", "-o", default=None,
                        help="Output directory for results (relative to script dir, "
                             "default: same as --in_dir)")
    parser.add_argument("--n_splits",     type=int, default=5)
    parser.add_argument("--all_concepts", action="store_true")
    parser.add_argument("--plot",         action="store_true")
    args = parser.parse_args()

    data_dir = Path(__file__).parent / args.in_dir
    out_dir  = Path(__file__).parent / (args.out_dir if args.out_dir else args.in_dir)
    out_dir.mkdir(exist_ok=True)
    stem     = args.stem
    prefix   = f"05_raw_{stem}"

    print(f"Loading: {stem}")
    resid_all = np.load(data_dir / f"01_{stem}_residual_all.npy")
    with open(data_dir / f"01_{stem}_meta.json") as f:
        meta = json.load(f)
    print(f"  Shape: {resid_all.shape}  |  N={len(meta)}")

    masks = _load_masks(data_dir, stem)

    ast_labels, builtin_labels, explicit_idxs = _get_explicit_labels(meta)
    print(f"  Explicit stubs (non-pseudo): {len(explicit_idxs)}")

    # -- Analysis 1: Shared subspace probe ------------------------------------
    shared_probe = None
    if masks:
        print("\n[1/3] Shared subspace probe...")
        shared_probe = run_shared_probe(
            resid_all, ast_labels, builtin_labels, explicit_idxs, masks, args.n_splits
        )
        acc   = shared_probe["accuracy"]
        folds = shared_probe["fold_scores"]

        np.savez_compressed(
            out_dir / f"{prefix}_shared_probe_accuracy.npz",
            shared_ast=acc["shared_ast"],
            shared_builtin=acc["shared_builtin"],
        )
        np.savez_compressed(
            out_dir / f"{prefix}_shared_probe_folds.npz",
            shared_ast=folds["shared_ast"],
            shared_builtin=folds["shared_builtin"],
        )
        print(f"  Saved {prefix}_shared_probe_accuracy.npz")
        print(f"\n  Shared probe (final layer):")
        print(f"    AST     : {acc['shared_ast'][-1]:.4f}  "
              f"(chance={shared_probe['chance_ast']:.4f})")
        print(f"    Builtin : {acc['shared_builtin'][-1]:.4f}  "
              f"(chance={shared_probe['chance_builtin']:.4f})")
    else:
        print("\n[1/3] Shared probe skipped (no masks).")

    # -- Analysis 2: Raw cross-probe ------------------------------------------
    print("\n[2/3] Raw cross-probe (full dims)...")
    raw_crossprobe = run_raw_crossprobe(
        resid_all, ast_labels, builtin_labels, explicit_idxs, args.n_splits
    )
    out_cp = out_dir / f"{prefix}_raw_crossprobe.json"
    with open(out_cp, "w") as f:
        json.dump(raw_crossprobe, f, indent=2)
    print(f"  Saved -> {out_cp}")

    last = raw_crossprobe[-1]
    print(f"\n  Raw cross-probe (final layer):")
    print(f"    Raw -> AST     : {last['raw_predict_ast']:.4f}  "
          f"(chance={last['chance_ast']:.4f})")
    print(f"    Raw -> Builtin : {last['raw_predict_builtin']:.4f}  "
          f"(chance={last['chance_builtin']:.4f})")

    # -- Analysis 3: Shared AUROC ---------------------------------------------
    ast_sh_auroc = {}
    b_sh_auroc   = {}
    if masks and args.all_concepts:
        print("\n[3/3] Shared subspace AUROC...")
        print("  AST concepts:")
        ast_sh_auroc = run_shared_auroc(resid_all, meta, masks, "ast")
        np.savez_compressed(
            out_dir / f"{prefix}_shared_auroc_ast.npz",
            **{k: v for k, v in ast_sh_auroc.items()}
        )
        print(f"  Saved {prefix}_shared_auroc_ast.npz ({len(ast_sh_auroc)} concepts)")

        print("  Builtin concepts:")
        b_sh_auroc = run_shared_auroc(resid_all, meta, masks, "builtin")
        np.savez_compressed(
            out_dir / f"{prefix}_shared_auroc_builtin.npz",
            **{k: v for k, v in b_sh_auroc.items()}
        )
        print(f"  Saved {prefix}_shared_auroc_builtin.npz ({len(b_sh_auroc)} concepts)")
    elif not args.all_concepts:
        print("\n[3/3] Shared AUROC skipped (pass --all_concepts to enable).")
    else:
        print("\n[3/3] Shared AUROC skipped (no masks).")

    # -- Comparison table -----------------------------------------------------
    pure_probe_path = data_dir / f"05_{stem}_probe_accuracy.npz"
    comparison = build_comparison(
        shared_probe,
        raw_crossprobe,
        pure_probe_path if pure_probe_path.exists() else None,
    )
    out_cmp = out_dir / f"{prefix}_shared_vs_pure_comparison.json"
    with open(out_cmp, "w") as f:
        json.dump(comparison, f, indent=2, default=float)
    print(f"\n  Saved comparison -> {out_cmp}")

    # -- Plots ----------------------------------------------------------------
    if args.plot:
        print("\nGenerating plots...")
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        plot_probe_comparison(
            comparison,
            save_path=img_dir / f"{prefix}_probe_comparison.png",
        )

        if args.all_concepts and masks:
            raw_ast_path = data_dir / f"05_{stem}_concept_auroc_ast.npz"
            raw_b_path   = data_dir / f"05_{stem}_concept_auroc_builtin.npz"

            if raw_ast_path.exists() and ast_sh_auroc:
                raw_ast = dict(np.load(raw_ast_path))
                plot_auroc_comparison(
                    raw_ast, ast_sh_auroc, "ast",
                    save_path=img_dir / f"{prefix}_auroc_ast_comparison.png",
                )
            if raw_b_path.exists() and b_sh_auroc:
                raw_b = dict(np.load(raw_b_path))
                plot_auroc_comparison(
                    raw_b, b_sh_auroc, "builtin",
                    save_path=img_dir / f"{prefix}_auroc_builtin_comparison.png",
                )

    print("\n[05_raw] Done.")


if __name__ == "__main__":
    main()
