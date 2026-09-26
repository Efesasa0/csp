"""
06_causal.py

Step 6 of the AST x builtin mechanistic interpretability pipeline.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY WE DO THIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Steps 02-05 established WHAT is encoded WHERE and HOW well.  They are all
correlational — a unit can covary with AST labels and still not *cause* any
AST-related behaviour.  Step 06 moves toward causality with three analyses:

  ANALYSIS A — OFFLINE ACTIVATION PATCHING
  -----------------------------------------
  True activation patching runs a new forward pass with some activations
  surgically replaced (requires the live model).  We approximate this using
  the linear superposition assumption of the residual stream:

      resid_patched[B] = resid[B] - resid_ast_component[B]
                                  + resid_ast_component[A]

  where:
    resid_ast_component[X] = resid_all[X] projected onto AST-pure dims (masked)
    A = source prompt (concept you want to "implant")
    B = target prompt (prompt you want to "patch into")

  This gives an approximation of "what would B's residual look like if we
  replaced its AST component with A's?"

  We then measure:
    patch_shift_to_A = cos_sim(resid_patched[B], resid[A]) - cos_sim(resid[B], resid[A])
    patch_shift_from_B = cos_sim(resid_patched[B], resid[B]) - 1.0

  Positive patch_shift_to_A → patched B is now closer to A → the AST-pure
  dims causally carry AST identity information (not just correlated with it).

  Note: this approximation is exact under perfect linear superposition and
  approximate otherwise.  It is valid as a first-order test without the model.

  ANALYSIS B — CAUSAL ABLATION IMPORTANCE
  -----------------------------------------
  For each subspace (ast_pure / builtin_pure / shared), ablate those
  dimensions in every prompt under two conditions and measure how much the
  final representation changes:

    zero ablation: set masked dims to 0
    mean ablation: set masked dims to the dataset-mean activation value

    ablation_effect[mode, subspace, prompt] = 1 - cos_sim(resid_ablated, resid_raw)

  Zero ablation is a stronger intervention (removes all signal including any
  baseline offset).  Mean ablation is softer — it only removes the
  prompt-specific deviation, preserving the average activation level.  If
  mean ablation has a much smaller effect than zero, the subspace carries a
  large constant offset.  If both are comparable, the subspace carries
  genuine discriminative signal.

  A large effect under either mode means the subspace is causally important
  for the overall representation.

  Plotted per layer and per category: we can see which subspace is most
  "load-bearing" at each depth.

  ANALYSIS C — MINIMAL CIRCUIT EXTRACTION
  -----------------------------------------
  Use the edge graph from step 01 + purity masks from step 02 to extract
  the minimal subgraph that implements AST or builtin processing:

    AST circuit    = edges where BOTH src and dst are ast_pure (or shared)
    Builtin circuit = edges where BOTH src and dst are builtin_pure (or shared)
    Interaction circuit = edges where src is ast_pure and dst is builtin_pure
                          (or vice versa) — these are the "crosswiring" edges

  For each edge in the circuit, we also report its VP-weighted importance
  score from step 02 (|weight| × unique_ast_score).

  The minimal circuit is the smallest set of edges that accounts for
  > threshold fraction of the total circuit importance score.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INPUT FILES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  01_<stem>_residual_all.npy     (N, L+1, H)
  01_<stem>_meta.json
  01_<stem>_edge_graph.npz       (optional, for Analysis C)
  02_<stem>_purity_masks.npz     (optional but strongly recommended)
  02_<stem>_vp_edge_weights.npz  (optional, for Analysis C importance scores)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FILES  (prefix 06_<stem>_)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  06_<stem>_patch_effects.npz        per-pair patching results (Analysis A)
  06_<stem>_patch_summary.json       mean patch shift per (AST, builtin) pair
  06_<stem>_ablation_effects.npz     per-subspace ablation effects (Analysis B)
  06_<stem>_ablation_summary.json    aggregated ablation stats
  06_<stem>_circuit_ast.json         minimal AST circuit edge list (Analysis C)
  06_<stem>_circuit_builtin.json     minimal builtin circuit edge list
  06_<stem>_circuit_interaction.json cross-wiring edges between circuits

Usage
-----
  python 06_causal.py --stem contrastive_stubs --plot
  python 06_causal.py --stem contrastive_stubs --no_circuit --plot
  python 06_causal.py --stem contrastive_stubs --circuit_threshold 0.80
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

COLOURS = {
    "ast_pure":     "#1565C0",
    "builtin_pure": "#E65100",
    "shared":       "#6A1B9A",
    "noise":        "#BDBDBD",
    "raw":          "#607D8B",
    "interaction":  "#C62828",
}


def _cos_sim_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity. A, B shape (N, D). Returns (N,)."""
    na = np.linalg.norm(A, axis=1, keepdims=True).clip(1e-12)
    nb = np.linalg.norm(B, axis=1, keepdims=True).clip(1e-12)
    return np.clip(((A / na) * (B / nb)).sum(axis=1), -1.0, 1.0)


def _load_masks(base: Path, stem: str) -> dict | None:
    path = base / f"02_{stem}_purity_masks.npz"
    if not path.exists():
        return None
    npz = np.load(path, allow_pickle=True)
    result = {}
    for key in ("ast_pure", "builtin_pure", "shared", "interaction", "noise"):
        fkey = f"residual_{key}"
        if fkey in npz.files:
            result[key] = npz[fkey].astype(bool)   # (L+1, H)
    return result if result else None


def _mask_at_layer(masks: dict, key: str, layer: int,
                   H: int) -> np.ndarray:
    """Return boolean mask of shape (H,) for a given key and layer."""
    m = masks.get(key)
    if m is None:
        return np.zeros(H, dtype=bool)
    return m[layer] if m.ndim == 2 else m


# ─────────────────────────────────────────────────────────────────────────────
# Analysis A — Offline activation patching
# ─────────────────────────────────────────────────────────────────────────────

def offline_patch_pairs(
    resid_all:  np.ndarray,    # (N, L+1, H)
    meta:       list[dict],
    masks:      dict | None,
    factor:     str = "ast",   # "ast" | "builtin"
    layer:      int = -1,
    max_pairs:  int = 500,
    seed:       int = 42,
) -> list[dict]:
    """
    Approximate activation patching for the chosen factor.

    For each pair (A, B) where A and B share the same OTHER factor but differ
    on `factor`:
      - Patch the `factor`-pure component of A into B
      - Measure how much patched-B shifts toward A in representation space

    Example with factor="ast":
      A, B share the same builtin but different AST nodes.
      Patch: replace B's ast_pure dims with A's ast_pure dims.
      If ast_pure dims causally carry AST identity, patched-B should look
      more like A than original B did.

    Returns list of dicts, one per pair.
    """
    L1 = resid_all.shape[1]
    l  = L1 + layer if layer < 0 else int(np.clip(layer, 0, L1 - 1))

    label_key_same  = "builtin_obj" if factor == "ast" else "ast_node"
    label_key_diff  = "ast_node"    if factor == "ast" else "builtin_obj"
    mask_key        = f"{factor}_pure"

    # Use only explicit stubs
    explicit_idx = [i for i, m in enumerate(meta)
                    if m.get("variant_type", "explicit") == "explicit"]

    # Group by "same" label
    groups: dict[str, list[int]] = defaultdict(list)
    for i in explicit_idx:
        groups[meta[i][label_key_same]].append(i)

    rng = np.random.default_rng(seed)
    pairs = []

    for same_label, idxs in groups.items():
        if len(idxs) < 2:
            continue
        # Create pairs with different `factor` labels
        diff_groups: dict[str, list[int]] = defaultdict(list)
        for i in idxs:
            diff_groups[meta[i][label_key_diff]].append(i)
        diff_labels = list(diff_groups.keys())
        if len(diff_labels) < 2:
            continue

        for _ in range(min(10, len(idxs))):
            la, lb = rng.choice(len(diff_labels), 2, replace=False)
            ia = rng.choice(diff_groups[diff_labels[la]])
            ib = rng.choice(diff_groups[diff_labels[lb]])
            pairs.append((int(ia), int(ib),
                          meta[ia][label_key_diff], meta[ib][label_key_diff],
                          same_label))

    # Subsample if too many
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    print(f"  Patching {len(pairs)} pairs (factor='{factor}', layer={l})...")

    act = resid_all[:, l, :]       # (N, H)
    H   = act.shape[1]

    if masks is not None:
        mask = _mask_at_layer(masks, mask_key, l, H)
    else:
        # Fallback: use all dims (no masking — approximation only)
        mask = np.ones(H, dtype=bool)
        print("  WARNING: no purity masks — patching ALL dims (rough approximation)")

    results = []
    for ia, ib, label_a, label_b, same_label in pairs:
        act_a = act[ia].copy()   # (H,)
        act_b = act[ib].copy()   # (H,)

        # Patch: replace B's factor dims with A's
        act_patched = act_b.copy()
        act_patched[mask] = act_a[mask]

        # Measure shifts
        cos_a_b     = float(_cos_sim_rows(act_a[None], act_b[None])[0])
        cos_a_patch = float(_cos_sim_rows(act_a[None], act_patched[None])[0])
        cos_b_patch = float(_cos_sim_rows(act_b[None], act_patched[None])[0])

        shift_toward_a  = cos_a_patch - cos_a_b    # did patched-B get closer to A?
        shift_from_b    = cos_b_patch - 1.0        # how much did B change? (<= 0)

        results.append({
            "source_idx":      ia,
            "target_idx":      ib,
            f"source_{factor}": label_a,
            f"target_{factor}": label_b,
            "shared_label":    same_label,
            "cos_A_B":         cos_a_b,
            "cos_A_patched":   cos_a_patch,
            "cos_B_patched":   cos_b_patch,
            "shift_toward_A":  shift_toward_a,
            "shift_from_B":    shift_from_b,
            "layer":           l,
            "factor":          factor,
            "n_patched_dims":  int(mask.sum()),
        })

    return results


def patch_all_layers(
    resid_all: np.ndarray,
    meta:      list[dict],
    masks:     dict | None,
    factor:    str = "ast",
    max_pairs: int = 200,
) -> tuple[list[dict], dict[int, list[dict]]]:
    """
    Run offline_patch_pairs at every layer.
    Returns (summary_list, pairs_by_layer) where:
      summary_list     — list of per-layer mean/std stats (for line plots)
      pairs_by_layer   — {layer_int: [raw pair dicts]} (for heatmaps)
    """
    N, L1, H = resid_all.shape
    all_results  = []
    pairs_by_layer: dict[int, list[dict]] = {}
    for l in range(L1):
        res = offline_patch_pairs(resid_all, meta, masks, factor, layer=l,
                                  max_pairs=max_pairs)
        pairs_by_layer[l] = res
        all_results.append({
            "layer":             l,
            "mean_shift_to_A":   float(np.mean([r["shift_toward_A"] for r in res])) if res else 0.0,
            "std_shift_to_A":    float(np.std( [r["shift_toward_A"] for r in res])) if res else 0.0,
            "mean_shift_from_B": float(np.mean([r["shift_from_B"]   for r in res])) if res else 0.0,
            "n_pairs":           len(res),
        })
        print(f"  Layer {l}: mean shift_toward_A={all_results[-1]['mean_shift_to_A']:.4f}", end="\r")
    print()
    return all_results, pairs_by_layer


# ─────────────────────────────────────────────────────────────────────────────
# Analysis B — Causal ablation importance
# ─────────────────────────────────────────────────────────────────────────────

def causal_ablation(
    resid_all: np.ndarray,    # (N, L+1, H)
    masks:     dict | None,
    subspaces: list[str] = ("ast_pure", "builtin_pure", "shared"),
) -> dict:
    """
    For each subspace, ablate those dimensions at every layer under two modes:
      zero — set masked dims to 0
      mean — set masked dims to the dataset-mean activation (per dim)

    Returns nested dict keyed by mode ("zero", "mean"), each containing:
      effects[subspace]     np.ndarray (L+1,)  mean cosine distance per layer
      effects_std[subspace] np.ndarray (L+1,)
      n_dims[subspace]      np.ndarray (L+1,)  number of dims ablated
    """
    N, L1, H = resid_all.shape
    modes = ("zero", "mean")
    results = {
        m: {
            "effects":     {s: np.zeros(L1) for s in subspaces},
            "effects_std": {s: np.zeros(L1) for s in subspaces},
            "n_dims":      {s: np.zeros(L1, dtype=int) for s in subspaces},
        }
        for m in modes
    }

    for l in range(L1):
        act_raw  = resid_all[:, l, :]          # (N, H)
        mean_act = act_raw.mean(axis=0)        # (H,)  dataset mean at this layer

        for s in subspaces:
            if masks is None:
                rng  = np.random.default_rng(42 + l)
                mask = rng.random(H) < 0.1
            else:
                mask = _mask_at_layer(masks, s, l, H)
                if s == "shared":
                    int_mask = _mask_at_layer(masks, "interaction", l, H)
                    mask = mask | int_mask

            n = int(mask.sum())
            results["zero"]["n_dims"][s][l] = n
            results["mean"]["n_dims"][s][l] = n

            # Zero ablation
            abl_zero = act_raw.copy()
            abl_zero[:, mask] = 0.0
            cs_zero = _cos_sim_rows(act_raw, abl_zero)
            results["zero"]["effects"][s][l]     = float(1.0 - cs_zero.mean())
            results["zero"]["effects_std"][s][l] = float(cs_zero.std())

            # Mean ablation
            abl_mean = act_raw.copy()
            abl_mean[:, mask] = mean_act[mask]
            cs_mean = _cos_sim_rows(act_raw, abl_mean)
            results["mean"]["effects"][s][l]     = float(1.0 - cs_mean.mean())
            results["mean"]["effects_std"][s][l] = float(cs_mean.std())

        print(f"  Ablation layer {l+1}/{L1}", end="\r")
    print()

    return results


def ablation_by_category(
    resid_all:      np.ndarray,
    meta:           list[dict],
    masks:          dict | None,
    ast_labels:     list,
    builtin_labels: list,
    subspace:       str = "ast_pure",
    layer:          int = -1,
    mode:           str = "zero",   # "zero" or "mean"
) -> tuple[dict, dict]:
    """
    Ablation effect broken down by AST category and builtin category.

    mode="zero" — set masked dims to 0
    mode="mean" — set masked dims to dataset-mean activation

    Returns (by_ast, by_builtin): dicts mapping label -> mean ablation effect.
    """
    L1 = resid_all.shape[1]
    l  = L1 + layer if layer < 0 else int(np.clip(layer, 0, L1 - 1))
    H  = resid_all.shape[2]

    if masks is not None:
        mask = _mask_at_layer(masks, subspace, l, H)
    else:
        mask = np.zeros(H, dtype=bool)

    act_raw     = resid_all[:, l, :]
    act_ablated = act_raw.copy()
    if mode == "mean":
        act_ablated[:, mask] = act_raw.mean(axis=0)[mask]
    else:
        act_ablated[:, mask] = 0.0
    cos_sim_vec = _cos_sim_rows(act_raw, act_ablated)  # (N,)
    effect_vec  = 1.0 - cos_sim_vec                    # (N,)

    by_ast: dict[str, float] = {}
    for label in sorted(set(ast_labels)):
        idx = [i for i, l_ in enumerate(ast_labels) if l_ == label]
        by_ast[label] = float(effect_vec[idx].mean())

    by_builtin: dict[str, float] = {}
    for label in sorted(set(builtin_labels)):
        idx = [i for i, l_ in enumerate(builtin_labels) if l_ == label]
        by_builtin[label] = float(effect_vec[idx].mean())

    return by_ast, by_builtin


# ─────────────────────────────────────────────────────────────────────────────
# Analysis C — Minimal circuit extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_circuits(
    edge_npz:           dict,     # loaded 01_*_edge_graph.npz
    vp_edge_npz:        dict | None,  # loaded 02_*_vp_edge_weights.npz
    masks:              dict | None,  # purity masks (L+1, H)
    circuit_threshold:  float = 0.80, # keep edges accounting for this fraction of importance
) -> dict:
    """
    Extract three circuits from the model's weight-sparsity edge graph:

      AST circuit:         src in ast_pure AND dst in ast_pure (or shared)
      Builtin circuit:     src in builtin_pure AND dst in builtin_pure (or shared)
      Interaction circuit: (src in ast_pure AND dst in builtin_pure) OR vice versa

    For each circuit, apply the threshold to keep only the most important edges
    (sorted by |weight| × VP score, or just |weight| if VP scores not available).

    Returns dict with "ast", "builtin", "interaction" circuits, each a list of
    edge dicts sorted by importance.
    """
    layers   = edge_npz.get("layer",    edge_npz.get("layers", np.array([]))).astype(int)
    src_idx  = edge_npz.get("src_idx",  np.array([])).astype(int)
    dst_idx  = edge_npz.get("dst_idx",  np.array([])).astype(int)
    weights  = edge_npz.get("weight",   edge_npz.get("weights", np.array([]))).astype(float)
    src_name = edge_npz.get("src_name", np.array([""] * len(weights), dtype=object))
    dst_name = edge_npz.get("dst_name", np.array([""] * len(weights), dtype=object))

    E = len(weights)
    if E == 0:
        print("  Edge graph is empty.")
        return {"ast": [], "builtin": [], "interaction": []}

    # VP-weighted importance (if available)
    if vp_edge_npz is not None and "edge_ast_importance" in vp_edge_npz:
        imp_ast = vp_edge_npz["edge_ast_importance"].astype(float)
        imp_b   = vp_edge_npz["edge_builtin_importance"].astype(float)
    else:
        imp_ast = np.abs(weights)
        imp_b   = np.abs(weights)

    # Build per-edge category based on src/dst purity
    if masks is not None:
        # Determine each unit's category at its layer
        def _unit_category(layer_i: int, unit_i: int) -> str:
            l = max(0, min(int(layer_i), masks["ast_pure"].shape[0] - 1))
            H = masks["ast_pure"].shape[1]
            u = min(int(unit_i), H - 1)
            if masks["ast_pure"][l, u]:
                return "ast_pure"
            if masks["builtin_pure"][l, u]:
                return "builtin_pure"
            sh  = masks.get("shared",      np.zeros_like(masks["ast_pure"]))
            ix  = masks.get("interaction", np.zeros_like(masks["ast_pure"]))
            if sh[l, u] or ix[l, u]:
                return "shared"
            return "noise"

        src_cats = np.array([_unit_category(layers[e], src_idx[e]) for e in range(E)],
                            dtype=object)
        dst_cats = np.array([_unit_category(layers[e], dst_idx[e]) for e in range(E)],
                            dtype=object)
    else:
        # No masks — all edges unclassified; return full graph
        src_cats = np.array(["unknown"] * E, dtype=object)
        dst_cats = np.array(["unknown"] * E, dtype=object)
        print("  No purity masks — returning full edge graph (unclassified).")

    def _filter_and_threshold(mask_bool: np.ndarray, importance: np.ndarray) -> list[dict]:
        """Keep edges in mask, sorted by importance, up to threshold fraction."""
        idx  = np.where(mask_bool)[0]
        if len(idx) == 0:
            return []
        imp  = importance[idx]
        order = np.argsort(imp)[::-1]
        idx   = idx[order]
        imp   = imp[order]

        cum_imp = np.cumsum(imp)
        total   = cum_imp[-1]
        if total > 0:
            cutoff = np.searchsorted(cum_imp, circuit_threshold * total)
            idx    = idx[: cutoff + 1]
        else:
            idx = idx

        return [
            {
                "src_name":  str(src_name[e]),
                "dst_name":  str(dst_name[e]),
                "src_idx":   int(src_idx[e]),
                "dst_idx":   int(dst_idx[e]),
                "layer":     int(layers[e]),
                "weight":    float(weights[e]),
                "importance":float(importance[e]),
                "src_cat":   str(src_cats[e]),
                "dst_cat":   str(dst_cats[e]),
            }
            for e in idx
        ]

    ast_mask   = np.isin(src_cats, ["ast_pure", "shared"]) & \
                 np.isin(dst_cats, ["ast_pure", "shared"])
    b_mask     = np.isin(src_cats, ["builtin_pure", "shared"]) & \
                 np.isin(dst_cats, ["builtin_pure", "shared"])
    inter_mask = (np.isin(src_cats, ["ast_pure"]) & np.isin(dst_cats, ["builtin_pure"])) | \
                 (np.isin(src_cats, ["builtin_pure"]) & np.isin(dst_cats, ["ast_pure"]))

    circuits = {
        "ast":         _filter_and_threshold(ast_mask,   imp_ast),
        "builtin":     _filter_and_threshold(b_mask,     imp_b),
        "interaction": _filter_and_threshold(inter_mask, (imp_ast + imp_b) / 2),
    }

    for name, edges in circuits.items():
        print(f"  {name:<15s} circuit: {len(edges)} edges "
              f"(threshold={circuit_threshold:.0%} of importance)")

    return circuits


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_patch_effects(
    patch_layer_results:  list[dict],
    factor:               str,
    save_path:            Path | None = None,
) -> None:
    """
    Line plot of mean activation patch shift across layers.
    shift_toward_A > 0 → patched prompt is closer to the source → causal.
    shift_from_B < 0 → target prompt changed after patching.

    Also shows a permutation-style null: expected shift if we patched random
    same-label pairs (shift_toward_A should be ~0 for within-category patches).
    """
    layers  = np.array([r["layer"]             for r in patch_layer_results])
    mean_a  = np.array([r["mean_shift_to_A"]   for r in patch_layer_results])
    std_a   = np.array([r["std_shift_to_A"]    for r in patch_layer_results])
    mean_b  = np.array([r["mean_shift_from_B"] for r in patch_layer_results])
    colour  = COLOURS["ast_pure"] if factor == "ast" else COLOURS["builtin_pure"]
    xtick_labels = ["emb" if l == 0 else str(l) for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    ax.fill_between(layers, mean_a - std_a, mean_a + std_a, alpha=0.2, color=colour)
    ax.plot(layers, mean_a, "o-", color=colour, lw=2,
            label="shift_toward_A (causal signal)")
    ax.axhline(0, color="grey", lw=1, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean shift toward source (cos_sim change)")
    ax.set_title(f"Patching {factor}-pure dims: does patched-B shift toward A?",
                 fontsize=10)
    ax.set_xticks(layers); ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(layers, mean_b, "s-", color="#607D8B", lw=2,
            label="shift_from_B (disruption to target)")
    ax.axhline(0, color="grey", lw=1, linestyle="--")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine sim change from original B")
    ax.set_title(f"How much does patching {factor}-pure dims disrupt target?",
                 fontsize=10)
    ax.set_xticks(layers); ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Offline activation patching — {factor} component\n"
        f"Positive shift_toward_A = {factor}-pure dims causally carry {factor} identity",
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_ablation_effects(
    ablation: dict,
    save_path: Path | None = None,
) -> None:
    """
    2×2 causal ablation plot.
      Top-left  — Zero ablation raw effect: how much does the representation
                  change when pure dims are set to zero?
      Top-right — Mean ablation raw effect: same but dims set to dataset mean
                  (softer intervention; large difference vs zero → constant offset).
      Bot-left  — Raw effect ÷ pure-dim count: removes the confound that bigger
                  subspaces always produce larger effects; shows causal importance
                  per individual dimension.
      Bot-right — How many pure dims each subspace has per layer (explains why
                  raw effects may differ between subspaces).
    Layer 0 is the raw embedding; x-tick labelled "emb".
    """
    subspaces = list(ablation["zero"]["effects"].keys())
    L1     = len(list(ablation["zero"]["effects"].values())[0])
    layers = np.arange(L1)
    xtick_labels = ["emb"] + [str(i) for i in range(1, L1)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()

    def _set_xticks(ax):
        ax.set_xticks(layers)
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_xlabel("Layer")

    # Top-left: zero ablation
    ax = axes[0]
    for s in subspaces:
        ax.plot(layers, ablation["zero"]["effects"][s], "o-",
                color=COLOURS.get(s, "#607D8B"), lw=2, label=s)
    ax.set_ylabel("Mean (1 − cos_sim) after ablation")
    ax.set_title("Zero ablation  [dims set to 0]\n"
                 "How much does removing this subspace distort the representation?",
                 fontsize=9)
    _set_xticks(ax); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Top-right: mean ablation
    ax = axes[1]
    for s in subspaces:
        ax.plot(layers, ablation["mean"]["effects"][s], "o--",
                color=COLOURS.get(s, "#607D8B"), lw=2, label=s)
    ax.set_ylabel("Mean (1 − cos_sim) after ablation")
    ax.set_title("Mean ablation  [dims set to dataset mean]\n"
                 "Softer: only removes prompt-specific deviation, not baseline offset",
                 fontsize=9)
    _set_xticks(ax); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Bottom-left: effect per pure dim
    ax = axes[2]
    for s in subspaces:
        n      = ablation["zero"]["n_dims"][s].astype(float)
        safe_n = np.where(n > 0, n, np.nan)
        ax.plot(layers, ablation["zero"]["effects"][s] / safe_n, "o-",
                color=COLOURS.get(s, "#607D8B"), lw=2, label=f"{s} (zero)")
        ax.plot(layers, ablation["mean"]["effects"][s] / safe_n, "o--",
                color=COLOURS.get(s, "#607D8B"), lw=1.5, alpha=0.6, label=f"{s} (mean)")
    ax.set_ylabel("(1 − cos_sim) per pure dim")
    ax.set_title("Causal importance per dimension  [effect ÷ pure-dim count]\n"
                 "Corrects for subspace size — fair comparison across subspaces",
                 fontsize=9)
    _set_xticks(ax); ax.legend(fontsize=7); ax.grid(True, alpha=0.2)

    # Bottom-right: pure dim count
    ax = axes[3]
    for s in subspaces:
        ax.plot(layers, ablation["zero"]["n_dims"][s], "o-",
                color=COLOURS.get(s, "#607D8B"), lw=1.5, label=s)
    ax.set_ylabel("Number of pure dims in subspace")
    ax.set_title("Subspace size per layer\n"
                 "Context: top-row effects scale with this, bottom-left corrects for it",
                 fontsize=9)
    _set_xticks(ax); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    fig.suptitle("Causal ablation: which subspace is most load-bearing?", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_ablation_by_category(
    by_ast:     dict[str, float],
    by_builtin: dict[str, float],
    subspace:   str,
    layer:      int,
    save_path:  Path | None = None,
) -> None:
    """
    Bar charts showing which AST nodes and builtins are most affected when
    the given subspace is ablated.  High ablation effect = that category
    relies heavily on this subspace.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, data, xlabel, colour in [
        (axes[0], by_ast,     "AST node",  COLOURS["ast_pure"]),
        (axes[1], by_builtin, "Builtin",   COLOURS["builtin_pure"]),
    ]:
        labels = sorted(data, key=lambda k: data[k], reverse=True)
        vals   = [data[k] for k in labels]
        x = np.arange(len(labels))
        ax.bar(x, vals, color=colour, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=max(5, 8 - len(labels) // 10))
        ax.set_ylabel("Mean ablation effect (1 - cos_sim)")
        ax.set_title(f"Effect of ablating '{subspace}' dims on each {xlabel}\n"
                     f"(layer {layer})", fontsize=10)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle(
        f"Which categories depend most on the '{subspace}' subspace?",
        fontsize=11
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_circuit_summary(
    circuits:  dict,    # from extract_circuits
    save_path: Path | None = None,
) -> None:
    """
    Three-panel circuit summary:
      Panel 1 — Edge count and total importance per circuit type
      Panel 2 — Distribution of edge importance scores per circuit (violin)
      Panel 3 — Layer distribution of edges per circuit (stacked bar)
    """
    names      = ["ast", "builtin", "interaction"]
    colours    = [COLOURS["ast_pure"], COLOURS["builtin_pure"], COLOURS["interaction"]]
    edge_lists = [circuits[n] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: counts and total importance
    ax = axes[0]
    counts = [len(e) for e in edge_lists]
    totals = [sum(ed["importance"] for ed in e) for e in edge_lists]
    x = np.arange(len(names))
    bars = ax.bar(x, counts, color=colours, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Number of edges in circuit")
    ax.set_title("Circuit sizes", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(x, totals, "D--", color="#333333", lw=1.5, markersize=7,
             label="Total importance")
    ax2.set_ylabel("Total VP-weighted importance")
    ax2.legend(fontsize=7, loc="upper right")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(c), ha="center", fontsize=10)

    # Panel 2: importance distribution (violin)
    ax = axes[1]
    data_for_violin = [
        [ed["importance"] for ed in e] for e in edge_lists if e
    ]
    valid_names   = [n for n, e in zip(names, edge_lists) if e]
    valid_colours = [c for c, e in zip(colours, edge_lists) if e]
    if data_for_violin:
        parts = ax.violinplot(data_for_violin, positions=range(len(valid_names)),
                              showmedians=True)
        for pc, col in zip(parts["bodies"], valid_colours):
            pc.set_facecolor(col)
            pc.set_alpha(0.7)
        ax.set_xticks(range(len(valid_names)))
        ax.set_xticklabels(valid_names, fontsize=9)
        ax.set_ylabel("Edge importance score")
        ax.set_title("Edge importance distribution per circuit", fontsize=10)

    # Panel 3: layer distribution
    ax = axes[2]
    if any(edge_lists):
        max_layer = max(
            (ed["layer"] for e in edge_lists for ed in e), default=0
        )
        layer_counts = np.zeros((len(names), max_layer + 1))
        for ni, e in enumerate(edge_lists):
            for ed in e:
                layer_counts[ni, ed["layer"]] += 1

        bottom = np.zeros(max_layer + 1)
        for ni, (name, col) in enumerate(zip(names, colours)):
            ax.bar(range(max_layer + 1), layer_counts[ni],
                   bottom=bottom, color=col, alpha=0.8, label=name)
            bottom += layer_counts[ni]
        ax.set_xlabel("Layer")
        ax.set_ylabel("Number of circuit edges")
        ax.set_title("Circuit edges by layer", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("Minimal circuit extraction summary\n"
                 "AST / builtin / interaction circuits from weight-sparsity graph",
                 fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {save_path}")
    plt.close(fig)


def _build_patch_grid(patch_results, src_key, tgt_key, src_labels, tgt_labels):
    """Aggregate patch_results into a (src × tgt) mean-shift grid."""
    grid   = np.full((len(src_labels), len(tgt_labels)), np.nan)
    counts = np.zeros_like(grid)
    for r in patch_results:
        i = src_labels.index(r[src_key])
        j = tgt_labels.index(r[tgt_key])
        if np.isnan(grid[i, j]):
            grid[i, j] = 0.0
        grid[i, j] += r["shift_toward_A"]
        counts[i, j] += 1
    return np.where(counts > 0, grid / counts, np.nan)


def _save_patch_heatmap_single(
    grid:       np.ndarray,
    src_labels: list[str],
    tgt_labels: list[str],
    factor:     str,
    title:      str,
    save_path:  Path,
) -> None:
    """Save one patch heatmap to its own file at LaTeX-ready resolution."""
    cmap = LinearSegmentedColormap.from_list(
        "GYR", ["#1a9641", "#FFFF33", "#d7191c"], N=256)
    cmap.set_bad("#d0d0d0")

    valid = grid[~np.isnan(grid)]
    if len(valid) == 0:
        return
    vmin = np.percentile(valid, 2)
    vmax = np.percentile(valid, 98)
    if vmin == vmax:
        vmin -= 1e-6; vmax += 1e-6

    fig_w = max(6, len(tgt_labels) * 0.55)
    fig_h = max(5, len(src_labels) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(grid, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Mean Δcos_sim toward source", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    ax.set_xticks(range(len(tgt_labels)))
    ax.set_xticklabels(tgt_labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticks(range(len(src_labels)))
    ax.set_yticklabels(src_labels, fontsize=11)
    ax.set_xlabel(f"Target {factor}", fontsize=12)
    ax.set_ylabel(f"Source {factor}", fontsize=12)
    ax.set_title(title, fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"  Saved -> {save_path}")
    plt.close(fig)


def plot_patch_heatmap(
    pairs_by_layer: dict[int, list[dict]],
    factor:         str,
    save_stem:      Path | None = None,
    fmt:            str = "pdf",
) -> None:
    """
    Save one heatmap file per layer plus one overall-average file.

    Files are named:  <save_stem>_L0.<fmt>, _L1.<fmt>, …, _avg.<fmt>

    Each file uses its own colour scale (2nd–98th percentile of that layer).
    Colourmap: green (low) → bright yellow → red (high causal signal).
    NaN cells are shown in light grey.
    """
    src_key = f"source_{factor}"
    tgt_key = f"target_{factor}"

    all_pairs = [r for pairs in pairs_by_layer.values() for r in pairs]
    if not all_pairs:
        return
    src_labels = sorted({r[src_key] for r in all_pairs})
    tgt_labels = sorted({r[tgt_key] for r in all_pairs})

    layers_sorted = sorted(pairs_by_layer)
    grids = {l: _build_patch_grid(pairs_by_layer[l], src_key, tgt_key,
                                  src_labels, tgt_labels)
             for l in layers_sorted}

    # Average grid (nanmean across layers)
    stack    = np.stack(list(grids.values()), axis=0)
    avg_grid = np.nanmean(stack, axis=0)
    avg_grid[np.all(np.isnan(stack), axis=0)] = np.nan

    panels = [(l, grids[l], "emb" if l == 0 else str(l))
              for l in layers_sorted] + [(-1, avg_grid, "avg")]

    for l, grid, lbl in panels:
        title = (f"Patch shift: source → target  ({factor}, layer {lbl})\n"
                 "Red = strong causal signal")
        if save_stem is not None:
            path = Path(str(save_stem) + f"_L{lbl}.{fmt}")
        else:
            path = None
        if path is not None:
            _save_patch_heatmap_single(
                grid, src_labels, tgt_labels, factor, title, path)


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 6: Offline patching, ablation, minimal circuit extraction"
    )
    parser.add_argument("--stem",               default="contrastive_stubs")
    parser.add_argument("--in_dir", "-d",       default="data",
                        help="Directory containing pipeline output files (default: data/)")
    parser.add_argument("--out_dir", "-o",      default=None,
                        help="Output directory for results (default: same as --in_dir)")
    parser.add_argument("--layer",              type=int, default=-1,
                        help="Layer for detailed analysis (-1 = final)")
    parser.add_argument("--max_pairs",          type=int, default=300,
                        help="Max pairs for patching per layer (default 300)")
    parser.add_argument("--circuit_threshold",  type=float, default=0.80,
                        help="Fraction of importance to retain in minimal circuit (default 0.80)")
    parser.add_argument("--no_patch",           action="store_true",
                        help="Skip Analysis A (activation patching)")
    parser.add_argument("--no_ablation",        action="store_true",
                        help="Skip Analysis B (causal ablation)")
    parser.add_argument("--no_circuit",         action="store_true",
                        help="Skip Analysis C (minimal circuit extraction)")
    parser.add_argument("--plot",               action="store_true")
    parser.add_argument("--fmt",                default="pdf",
                        choices=["pdf", "png"],
                        help="Output format for figures (default: pdf)")
    args = parser.parse_args()

    base    = Path(__file__).parent / args.in_dir
    out     = Path(__file__).parent / (args.out_dir if args.out_dir else args.in_dir)
    out.mkdir(exist_ok=True)
    img_dir = out / "images"
    img_dir.mkdir(exist_ok=True)
    stem = args.stem

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"\n[06] Loading inputs for stem='{stem}'")
    resid_all = np.load(base / f"01_{stem}_residual_all.npy")
    N, L1, H  = resid_all.shape
    with open(base / f"01_{stem}_meta.json") as f:
        meta = json.load(f)
    ast_labels     = [m["ast_node"]    for m in meta]
    builtin_labels = [m["builtin_obj"] for m in meta]
    print(f"  Shape: {resid_all.shape}")

    masks = _load_masks(base, stem)
    if masks is None:
        print("  No purity masks — analyses will use all dims (rough).")
    else:
        print("  Purity masks loaded.")

    layer_idx = L1 + args.layer if args.layer < 0 else args.layer
    layer_idx = int(np.clip(layer_idx, 0, L1 - 1))

    # ── Analysis A: Activation patching ──────────────────────────────────────
    ast_patch_layers   = []
    b_patch_layers     = []
    ast_pairs_by_layer: dict[int, list[dict]] = {}

    if not args.no_patch:
        print(f"\n[06] Analysis A: Offline activation patching (all layers)...")
        for factor in ("ast", "builtin"):
            print(f"\n  Factor: {factor}")
            layer_results, pairs_by_layer = patch_all_layers(
                resid_all, meta, masks, factor=factor, max_pairs=args.max_pairs
            )
            if factor == "ast":
                ast_patch_layers   = layer_results
                ast_pairs_by_layer = pairs_by_layer
            else:
                b_patch_layers = layer_results

        # Save
        patch_summary = {
            "ast":     [{"layer": r["layer"],
                          "mean_shift": r["mean_shift_to_A"],
                          "std_shift":  r["std_shift_to_A"]}
                        for r in ast_patch_layers],
            "builtin": [{"layer": r["layer"],
                          "mean_shift": r["mean_shift_to_A"],
                          "std_shift":  r["std_shift_to_A"]}
                        for r in b_patch_layers],
        }
        with open(out / f"06_{stem}_patch_summary.json", "w") as f:
            json.dump(patch_summary, f, indent=2)
        print(f"\n  Saved 06_{stem}_patch_summary.json")

    # ── Analysis B: Causal ablation ───────────────────────────────────────────
    ablation = {}
    by_ast_abl, by_b_abl = {}, {}

    if not args.no_ablation:
        print(f"\n[06] Analysis B: Causal ablation — zero + mean (all layers)...")
        ablation = causal_ablation(resid_all, masks)   # {"zero": {...}, "mean": {...}}

        # Per-category ablation at chosen layer (both modes)
        print(f"\n  Per-category ablation at layer {layer_idx} (ast_pure)...")
        by_ast_abl, by_b_abl = {}, {}
        for mode in ("zero", "mean"):
            ba, bb = ablation_by_category(
                resid_all, meta, masks, ast_labels, builtin_labels,
                subspace="ast_pure", layer=layer_idx, mode=mode,
            )
            by_ast_abl[mode] = ba
            by_b_abl[mode]   = bb

        abl_save = {}
        for mode in ("zero", "mean"):
            for s in ablation[mode]["effects"]:
                abl_save[f"effect_{mode}_{s}"] = ablation[mode]["effects"][s]
                abl_save[f"ndims_{s}"]         = ablation[mode]["n_dims"][s].astype(float)
        np.savez_compressed(out / f"06_{stem}_ablation_effects.npz", **abl_save)

        abl_summary = {
            mode: {
                "subspaces": {
                    s: {
                        "mean_effect_final_layer": float(ablation[mode]["effects"][s][layer_idx]),
                        "mean_effect_all_layers":  float(ablation[mode]["effects"][s].mean()),
                        "peak_layer":              int(ablation[mode]["effects"][s].argmax()),
                    }
                    for s in ablation[mode]["effects"]
                },
                "by_ast_at_final_layer":     by_ast_abl[mode],
                "by_builtin_at_final_layer": by_b_abl[mode],
            }
            for mode in ("zero", "mean")
        }
        with open(out / f"06_{stem}_ablation_summary.json", "w") as f:
            json.dump(abl_summary, f, indent=2)
        print(f"  Saved ablation outputs.")

    # ── Analysis C: Minimal circuit extraction ────────────────────────────────
    circuits = {}

    if not args.no_circuit:
        edge_path = base / f"01_{stem}_edge_graph.npz"
        if not edge_path.exists():
            print(f"\n[06] Edge graph not found ({edge_path.name}) — skipping circuit extraction.")
            print("  Run 01_extraction.py without --skip_edge_graph first.")
        else:
            print(f"\n[06] Analysis C: Minimal circuit extraction...")
            edge_npz = dict(np.load(edge_path, allow_pickle=True))

            vp_edge_path = base / f"02_{stem}_vp_edge_weights.npz"
            vp_edge_npz  = dict(np.load(vp_edge_path, allow_pickle=True)) \
                           if vp_edge_path.exists() else None

            circuits = extract_circuits(
                edge_npz, vp_edge_npz, masks,
                circuit_threshold=args.circuit_threshold
            )

            for name, edges in circuits.items():
                path = out / f"06_{stem}_circuit_{name}.json"
                with open(path, "w") as f:
                    json.dump(edges, f, indent=2)
                print(f"  Saved {path.name}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    if args.plot:
        print(f"\n[06] Generating plots...")

        if ast_patch_layers:
            plot_patch_effects(
                ast_patch_layers, factor="ast",
                save_path=img_dir / f"06_{stem}_patch_ast_layers.png",
            )
        if b_patch_layers:
            plot_patch_effects(
                b_patch_layers, factor="builtin",
                save_path=img_dir / f"06_{stem}_patch_builtin_layers.png",
            )
        if ast_pairs_by_layer:
            plot_patch_heatmap(
                ast_pairs_by_layer, factor="ast",
                save_stem=img_dir / f"06_{stem}_patch_ast_heatmap",
                fmt=args.fmt,
            )

        if ablation:
            plot_ablation_effects(
                ablation,
                save_path=img_dir / f"06_{stem}_ablation_effects.png",
            )
            if by_ast_abl or by_b_abl:
                for mode in ("zero", "mean"):
                    plot_ablation_by_category(
                        by_ast_abl.get(mode, {}), by_b_abl.get(mode, {}),
                        subspace="ast_pure", layer=layer_idx,
                        save_path=img_dir / f"06_{stem}_ablation_by_category_{mode}.png",
                    )

        if circuits:
            plot_circuit_summary(
                circuits,
                save_path=img_dir / f"06_{stem}_circuit_summary.png",
            )

        print(f"  All plots saved to {out}/")

    print("\n[06] Done.")


if __name__ == "__main__":
    main()
