#!/usr/bin/env python3
"""
X06 Phase B — Subspace geometry analysis (offline, no GPU required)

Reads the intermediate HDF5 produced by x06_extract_subspace.py and runs
the full geometry analysis: centroids, reconstruction models, interaction
residuals, linear probes, PCA, and neighborhood distances.

Usage:
    python experiments/scripts/x06_subspace_geometry.py \
        --acts experiments/outputs/x06/For__list_activations.h5 \
        --out experiments/outputs/x06 \
        --probe-cv 5 --pca-components 50

Requires: numpy, pandas, h5py, matplotlib, scikit-learn
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

ROOT = Path(__file__).parents[2]


def parse_args():
    p = argparse.ArgumentParser(
        description="X06-B: subspace geometry analysis")
    p.add_argument("--acts",
                   default="experiments/outputs/x06/activations.h5")
    p.add_argument("--out", default="experiments/outputs/x06")
    p.add_argument("--probe-cv", type=int, default=5,
                   help="Cross-validation folds for linear probes")
    p.add_argument("--pca-components", type=int, default=50,
                   help="Number of PCA components to compute")
    p.add_argument("--full-probes", action="store_true",
                   help="Run probes on full features (slow, default: PCA-100)")
    p.add_argument("--skip-steering", action="store_true",
                   help="Skip causal steering analysis (not yet implemented)")
    return p.parse_args()


def resolve(path: str) -> Path:
    pp = Path(path)
    return pp if pp.is_absolute() else ROOT / pp


# ── Data loading ─────────────────────────────────────────────────────────

def load_activations(acts_path: Path) -> tuple[dict, pd.DataFrame, dict]:
    """Load HDF5 and return metadata, prompts DataFrame, and activations dict."""
    def decode(arr):
        return [x.decode() if isinstance(x, bytes) else x for x in arr]

    with h5py.File(str(acts_path), "r") as f:
        meta = {
            "model_id": f.attrs["model_id"],
            "n_layers": int(f.attrs["n_layers"]),
            "n_neurons": int(f.attrs["n_neurons"]),
            "target_ast": f.attrs["target_ast"],
            "target_builtin": f.attrs["target_builtin"],
        }

        prompts = pd.DataFrame({
            "text": decode(f["prompts/text"][:]),
            "ast_node": decode(f["prompts/ast_node"][:]),
            "builtin_obj": decode(f["prompts/builtin_obj"][:]),
            "group": decode(f["prompts/group"][:]),
            "variation_id": f["prompts/variation_id"][:],
        })

        activations = {}
        for lid in range(meta["n_layers"]):
            key = f"activations/layer_{lid}"
            if key in f:
                activations[lid] = f[key][:].astype(np.float32)

    return meta, prompts, activations


# ── Step 1-2: Centroids ─────────────────────────────────────────────────

def compute_centroids(acts: np.ndarray, prompts: pd.DataFrame) -> dict:
    """
    Compute centroids for each group type.

    Returns dict with keys:
        global_mean, ast_centroids, builtin_centroids, pair_centroids
    """
    global_mean = acts.mean(axis=0)

    ast_centroids = {}
    for ast_n in prompts["ast_node"].unique():
        idx = prompts["ast_node"] == ast_n
        ast_centroids[ast_n] = acts[idx.values].mean(axis=0)

    builtin_centroids = {}
    for blt in prompts["builtin_obj"].unique():
        idx = prompts["builtin_obj"] == blt
        builtin_centroids[blt] = acts[idx.values].mean(axis=0)

    pair_centroids = {}
    for (ast_n, blt), group in prompts.groupby(["ast_node", "builtin_obj"]):
        pair_centroids[(ast_n, blt)] = acts[group.index.values].mean(axis=0)

    return {
        "global_mean": global_mean,
        "ast_centroids": ast_centroids,
        "builtin_centroids": builtin_centroids,
        "pair_centroids": pair_centroids,
    }


# ── Step 3-4: Reconstruction metrics ────────────────────────────────────

def reconstruction_metrics(
    acts: np.ndarray,
    prompts: pd.DataFrame,
    centroids: dict,
) -> pd.DataFrame:
    """
    Compute R² and cosine similarity for 4 reconstruction models.

    Models:
        A (AST-only):     h_hat = ast_centroid[a]
        B (builtin-only): h_hat = builtin_centroid[b]
        C (additive):     h_hat = global_mean + (ast_centroid[a] - global_mean)
                                              + (builtin_centroid[b] - global_mean)
        D (pair-specific): h_hat = pair_centroid[(a,b)]
    """
    gm = centroids["global_mean"]
    rows = []
    skip_centroid = 0
    skip_sstot = 0

    for i in range(len(acts)):
        h = acts[i]
        ast_n = prompts.iloc[i]["ast_node"]
        blt = prompts.iloc[i]["builtin_obj"]
        group = prompts.iloc[i]["group"]

        ast_c = centroids["ast_centroids"].get(ast_n)
        blt_c = centroids["builtin_centroids"].get(blt)
        pair_c = centroids["pair_centroids"].get((ast_n, blt))

        if ast_c is None or blt_c is None or pair_c is None:
            skip_centroid += 1
            continue

        # Reconstruction predictions
        h_hat_A = ast_c
        h_hat_B = blt_c
        h_hat_C = gm + (ast_c - gm) + (blt_c - gm)
        h_hat_D = pair_c

        # Denominators
        ss_tot = np.sum((h - gm) ** 2)
        if ss_tot < 1e-30:  # very loose — float16 data can have small magnitudes
            skip_sstot += 1
            continue

        for model_name, h_hat in [("A_ast", h_hat_A), ("B_builtin", h_hat_B),
                                   ("C_additive", h_hat_C), ("D_pair", h_hat_D)]:
            ss_res = np.sum((h - h_hat) ** 2)
            r2 = 1.0 - ss_res / ss_tot

            # Cosine similarity
            norm_h = np.linalg.norm(h)
            norm_hh = np.linalg.norm(h_hat)
            if norm_h > 0 and norm_hh > 0:
                cos = float(np.dot(h, h_hat) / (norm_h * norm_hh))
            else:
                cos = 0.0

            rows.append({
                "prompt_idx": i,
                "ast_node": ast_n,
                "builtin_obj": blt,
                "group": group,
                "model": model_name,
                "r2": float(r2),
                "cosine": cos,
            })

    if skip_centroid or skip_sstot:
        print(f"    reconstruction_metrics: {len(rows)//4} prompts used, "
              f"{skip_centroid} skipped (missing centroid), "
              f"{skip_sstot} skipped (ss_tot≈0)")

    return pd.DataFrame(rows)


# ── Step 5: Interaction residual ─────────────────────────────────────────

def interaction_residual(
    centroids: dict,
    target_ast: str,
    target_builtin: str,
) -> dict:
    """
    Compute pair_centroid - additive_reconstruction for the target pair.

    Returns dict with residual vector, its norm, and the additive reconstruction.
    """
    gm = centroids["global_mean"]
    ast_c = centroids["ast_centroids"][target_ast]
    blt_c = centroids["builtin_centroids"][target_builtin]
    pair_c = centroids["pair_centroids"][(target_ast, target_builtin)]

    additive = gm + (ast_c - gm) + (blt_c - gm)
    residual = pair_c - additive

    return {
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "additive": additive,
        "pair_centroid": pair_c,
    }


# ── Step 6: Pair stability ──────────────────────────────────────────────

def pair_stability(
    acts: np.ndarray,
    prompts: pd.DataFrame,
    centroids: dict,
) -> pd.DataFrame:
    """
    Within-pair variance vs between-pair centroid distance.
    """
    rows = []
    pair_centroids = centroids["pair_centroids"]

    for (ast_n, blt), group in prompts.groupby(["ast_node", "builtin_obj"]):
        idx = group.index.values
        if len(idx) < 2:
            continue
        pair_acts = acts[idx]
        pc = pair_centroids[(ast_n, blt)]

        # Within-pair: mean distance to own centroid
        dists_within = np.linalg.norm(pair_acts - pc, axis=1)
        within_mean = float(dists_within.mean())

        # Between-pair: distance to other pair centroids
        between_dists = []
        for other_key, other_c in pair_centroids.items():
            if other_key != (ast_n, blt):
                between_dists.append(float(np.linalg.norm(pc - other_c)))

        between_mean = float(np.mean(between_dists)) if between_dists else 0.0

        rows.append({
            "ast_node": ast_n,
            "builtin_obj": blt,
            "pair": f"{ast_n}__{blt}",
            "n_prompts": len(idx),
            "within_pair_dist": within_mean,
            "between_pair_dist": between_mean,
            "ratio": between_mean / within_mean if within_mean > 0 else float("inf"),
        })

    return pd.DataFrame(rows)


# ── Step 7: Linear probes ───────────────────────────────────────────────

def _reduce_dims(acts: np.ndarray, max_dims: int = 100) -> np.ndarray:
    """PCA reduction to speed up probes when n_features >> n_samples."""
    if acts.shape[1] <= max_dims:
        return acts
    centered = acts - acts.mean(axis=0)
    n_comp = min(max_dims, *centered.shape)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_comp].T


def train_probes(
    acts: np.ndarray,
    prompts: pd.DataFrame,
    n_cv: int,
    full: bool = False,
) -> dict:
    """
    Train logistic regression probes for AST, builtin, and pair labels.

    Returns dict of {label_type: accuracy}.
    """
    if full:
        acts_reduced = acts
        solver, max_iter, tol = "lbfgs", 1000, 1e-4
    else:
        acts_reduced = _reduce_dims(acts)
        solver, max_iter, tol = "saga", 500, 1e-3
    results = {}

    for label_col, label_name in [
        ("ast_node", "ast"),
        ("builtin_obj", "builtin"),
    ]:
        labels = prompts[label_col].values
        le = LabelEncoder()
        y = le.fit_transform(labels)

        if len(np.unique(y)) < 2:
            results[label_name] = float("nan")
            continue

        n_folds = min(n_cv, min(np.bincount(y)))
        if n_folds < 2:
            results[label_name] = float("nan")
            continue

        clf = LogisticRegression(max_iter=max_iter, solver=solver,
                                 multi_class="multinomial", tol=tol)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(clf, acts_reduced, y, cv=cv,
                                     scoring="accuracy")
        results[label_name] = float(scores.mean())

    # Pair label
    pair_labels = prompts["ast_node"] + "__" + prompts["builtin_obj"]
    le = LabelEncoder()
    y = le.fit_transform(pair_labels.values)

    if len(np.unique(y)) >= 2:
        n_folds = min(n_cv, min(np.bincount(y)))
        if n_folds >= 2:
            clf = LogisticRegression(max_iter=500, solver="saga",
                                     multi_class="multinomial", tol=1e-3)
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = cross_val_score(clf, acts_reduced, y, cv=cv,
                                         scoring="accuracy")
            results["pair"] = float(scores.mean())
        else:
            results["pair"] = float("nan")
    else:
        results["pair"] = float("nan")

    return results


# ── Step 8: PCA analysis ────────────────────────────────────────────────

def pca_analysis(acts: np.ndarray, n_components: int) -> dict:
    """
    PCA on activations: explained variance, effective rank, participation ratio.
    """
    centered = acts - acts.mean(axis=0)
    n_comp = min(n_components, min(centered.shape))

    # SVD-based PCA (more numerically stable for wide matrices)
    _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    all_eigenvalues = (S ** 2) / (len(acts) - 1)
    eigenvalues = all_eigenvalues[:n_comp]

    # total_var uses the FULL spectrum so ratios are relative to all variance
    total_var = float(all_eigenvalues.sum())
    if total_var > 0:
        explained_ratio = eigenvalues / total_var
    else:
        explained_ratio = np.zeros(n_comp)

    cumulative = np.cumsum(explained_ratio)

    # Effective rank (Shannon entropy-based) — uses full spectrum
    p_full = all_eigenvalues / total_var if total_var > 0 else np.ones(len(all_eigenvalues)) / len(all_eigenvalues)
    p_full = p_full[p_full > 1e-12]
    effective_rank = float(np.exp(-np.sum(p_full * np.log(p_full))))

    # Participation ratio — uses full spectrum
    if total_var > 0:
        participation_ratio = float(total_var ** 2 / np.sum(all_eigenvalues ** 2))
    else:
        participation_ratio = 0.0

    # Top-2 PCA coordinates for visualization
    pca_2d = centered @ Vt[:2].T  # (N, 2)

    return {
        "explained_ratio": explained_ratio,
        "cumulative": cumulative,
        "eigenvalues": eigenvalues,
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "pca_2d": pca_2d,
        "components": Vt[:n_comp],
    }


# ── Step 9: Neighborhood distances ──────────────────────────────────────

def neighborhood_distances(centroids: dict) -> pd.DataFrame:
    """Pairwise centroid distances for all pairs."""
    pair_centroids = centroids["pair_centroids"]
    keys = sorted(pair_centroids.keys(), key=lambda x: f"{x[0]}__{x[1]}")

    rows = []
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            if i >= j:
                continue
            d = float(np.linalg.norm(pair_centroids[k1] - pair_centroids[k2]))
            rows.append({
                "pair_a": f"{k1[0]}__{k1[1]}",
                "pair_b": f"{k2[0]}__{k2[1]}",
                "distance": d,
            })

    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_model_comparison(all_recon: pd.DataFrame, out_dir: Path, n_layers: int):
    """R² and cosine for 4 models across layers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax in zip(["r2", "cosine"], axes):
        for model in ["A_ast", "B_builtin", "C_additive", "D_pair"]:
            sub = all_recon[all_recon["model"] == model]
            means = sub.groupby("layer")[metric].mean()
            ax.plot(means.index, means.values, marker="o", label=model)
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"Reconstruction {metric.upper()} by layer")
        ax.legend()
        ax.set_xticks(range(n_layers))

    fig.tight_layout()
    fig.savefig(out_dir / "model_comparison_by_layer.png", dpi=120)
    plt.close(fig)


def plot_interaction_residual(residuals: list[dict], out_dir: Path, n_layers: int):
    """||interaction residual|| across layers."""
    layers = [r["layer"] for r in residuals]
    norms = [r["residual_norm"] for r in residuals]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(layers, norms, color="coral")
    ax.set_xlabel("Layer")
    ax.set_ylabel("||Interaction Residual||")
    ax.set_title("Interaction residual norm by layer\n"
                 "(low = compositional, high = entangled)")
    ax.set_xticks(range(n_layers))
    fig.tight_layout()
    fig.savefig(out_dir / "interaction_residual_by_layer.png", dpi=120)
    plt.close(fig)


def plot_pair_stability(all_stability: pd.DataFrame, out_dir: Path, n_layers: int):
    """Within vs between distances by layer."""
    target_stab = all_stability.groupby("layer").agg(
        within=("within_pair_dist", "mean"),
        between=("between_pair_dist", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(target_stab["layer"], target_stab["within"],
            marker="o", label="Within-pair dist")
    ax.plot(target_stab["layer"], target_stab["between"],
            marker="s", label="Between-pair dist")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Euclidean Distance")
    ax.set_title("Pair stability: within vs between distances")
    ax.legend()
    ax.set_xticks(range(n_layers))
    fig.tight_layout()
    fig.savefig(out_dir / "pair_stability_by_layer.png", dpi=120)
    plt.close(fig)


def plot_probe_accuracy(all_probes: pd.DataFrame, out_dir: Path, n_layers: int):
    """AST/builtin/pair probe accuracy by layer."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label_type in ["ast", "builtin", "pair"]:
        sub = all_probes[all_probes["label_type"] == label_type]
        ax.plot(sub["layer"], sub["accuracy"], marker="o", label=label_type)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance (binary)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Linear probe accuracy by layer")
    ax.legend()
    ax.set_xticks(range(n_layers))
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_dir / "probe_accuracy_by_layer.png", dpi=120)
    plt.close(fig)


def plot_pca_explained_variance(all_pca: dict, out_dir: Path, n_layers: int):
    """Explained variance per layer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative variance for each layer
    ax = axes[0]
    for lid in sorted(all_pca.keys()):
        cum = all_pca[lid]["cumulative"]
        ax.plot(range(1, len(cum) + 1), cum, label=f"L{lid}", alpha=0.7)
    ax.set_xlabel("# Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA cumulative explained variance")
    ax.legend(fontsize=7)

    # Effective rank and participation ratio
    ax = axes[1]
    layers = sorted(all_pca.keys())
    eff_ranks = [all_pca[l]["effective_rank"] for l in layers]
    part_ratios = [all_pca[l]["participation_ratio"] for l in layers]
    ax.plot(layers, eff_ranks, marker="o", label="Effective rank")
    ax.plot(layers, part_ratios, marker="s", label="Participation ratio")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Value")
    ax.set_title("Dimensionality measures by layer")
    ax.legend()
    ax.set_xticks(range(n_layers))

    fig.tight_layout()
    fig.savefig(out_dir / "pca_explained_variance.png", dpi=120)
    plt.close(fig)


def plot_pca_2d(pca_2d: np.ndarray, prompts: pd.DataFrame,
                layer: int, out_dir: Path):
    """2D PCA scatter colored by group."""
    groups = prompts["group"].values
    unique_groups = sorted(set(groups))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_groups), 1)))

    fig, ax = plt.subplots(figsize=(8, 6))
    for gi, g in enumerate(unique_groups):
        mask = groups == g
        ax.scatter(pca_2d[mask, 0], pca_2d[mask, 1],
                   c=[colors[gi]], label=g, alpha=0.6, s=30)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"PCA 2D — Layer {layer}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"pca_2d_layer{layer}.png", dpi=120)
    plt.close(fig)


def plot_neighborhood_heatmap(all_nbr: pd.DataFrame, out_dir: Path):
    """Centroid distance heatmap (average across layers or per-layer)."""
    if all_nbr.empty:
        return

    # Average across layers
    avg = all_nbr.groupby(["pair_a", "pair_b"])["distance"].mean().reset_index()
    pairs_u = sorted(set(avg["pair_a"]) | set(avg["pair_b"]))
    n = len(pairs_u)
    if n < 2:
        return

    idx_map = {p: i for i, p in enumerate(pairs_u)}
    matrix = np.zeros((n, n))
    for _, row in avg.iterrows():
        i, j = idx_map[row["pair_a"]], idx_map[row["pair_b"]]
        matrix[i, j] = row["distance"]
        matrix[j, i] = row["distance"]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_xticks(range(n))
    ax.set_xticklabels(pairs_u, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(pairs_u, fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean centroid distance")
    ax.set_title("Pair centroid distances (averaged across layers)")
    fig.tight_layout()
    fig.savefig(out_dir / "neighborhood_distances.png", dpi=120)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    acts_path = resolve(args.acts)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not acts_path.exists():
        print(f"ERROR: activation file not found: {acts_path}")
        print("Run x06_extract_subspace.py first.")
        sys.exit(1)

    print(f"Loading activations: {acts_path}")
    meta, prompts, activations = load_activations(acts_path)
    n_layers = meta["n_layers"]
    target_ast = meta["target_ast"]
    target_builtin = meta["target_builtin"]
    print(f"  Target: ({target_ast}, {target_builtin})")
    print(f"  {len(prompts)} prompts, {n_layers} layers, "
          f"groups: {prompts['group'].value_counts().to_dict()}")

    # Accumulators for cross-layer results
    all_recon_rows = []
    all_stability_rows = []
    all_probe_rows = []
    all_pca = {}
    all_nbr_rows = []
    all_residuals = []

    STEPS = ["centroids", "reconstruction", "interaction", "stability",
             "probes", "PCA", "neighborhood"]

    layer_bar = tqdm(range(n_layers), desc="Layers", unit="layer")
    for lid in layer_bar:
        acts = activations.get(lid)
        if acts is None:
            continue

        step_bar = tqdm(STEPS, desc=f"  L{lid}", leave=False, unit="step")

        # 1-2. Centroids
        step_bar.set_description(f"  L{lid} centroids")
        centroids = compute_centroids(acts, prompts)
        step_bar.update(1)

        # 3-4. Reconstruction metrics
        step_bar.set_description(f"  L{lid} reconstruction")
        recon_df = reconstruction_metrics(acts, prompts, centroids)
        if not recon_df.empty:
            recon_df["layer"] = lid
            all_recon_rows.append(recon_df)
        step_bar.update(1)

        # 5. Interaction residual
        step_bar.set_description(f"  L{lid} interaction")
        if (target_ast in centroids["ast_centroids"] and
                target_builtin in centroids["builtin_centroids"] and
                (target_ast, target_builtin) in centroids["pair_centroids"]):
            ir = interaction_residual(centroids, target_ast, target_builtin)
            ir["layer"] = lid
            all_residuals.append(ir)
        step_bar.update(1)

        # 6. Pair stability
        step_bar.set_description(f"  L{lid} stability")
        stab_df = pair_stability(acts, prompts, centroids)
        stab_df["layer"] = lid
        all_stability_rows.append(stab_df)
        step_bar.update(1)

        # 7. Linear probes
        step_bar.set_description(f"  L{lid} probes")
        probe_results = train_probes(acts, prompts, n_cv=args.probe_cv,
                                     full=args.full_probes)
        for lt, acc in probe_results.items():
            all_probe_rows.append({"layer": lid, "label_type": lt, "accuracy": acc})
        step_bar.update(1)

        # 8. PCA
        step_bar.set_description(f"  L{lid} PCA")
        pca = pca_analysis(acts, n_components=args.pca_components)
        all_pca[lid] = pca
        step_bar.update(1)

        # 9. Neighborhood distances
        step_bar.set_description(f"  L{lid} neighborhood")
        nbr_df = neighborhood_distances(centroids)
        nbr_df["layer"] = lid
        all_nbr_rows.append(nbr_df)
        step_bar.update(1)

        step_bar.close()

    # ── Combine and save ─────────────────────────────────────────────────

    all_recon = pd.concat(all_recon_rows, ignore_index=True) if all_recon_rows else pd.DataFrame()
    all_stability = pd.concat(all_stability_rows, ignore_index=True) if all_stability_rows else pd.DataFrame()
    all_probes = pd.DataFrame(all_probe_rows)
    all_nbr = pd.concat(all_nbr_rows, ignore_index=True) if all_nbr_rows else pd.DataFrame()

    # CSV
    csv_path = out_dir / "reconstruction_metrics.csv"
    all_recon.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    stab_csv = out_dir / "pair_stability.csv"
    all_stability.to_csv(stab_csv, index=False)
    print(f"Saved: {stab_csv}")

    probe_csv = out_dir / "probe_accuracy.csv"
    all_probes.to_csv(probe_csv, index=False)
    print(f"Saved: {probe_csv}")

    if not all_nbr.empty:
        nbr_csv = out_dir / "neighborhood_distances.csv"
        all_nbr.to_csv(nbr_csv, index=False)
        print(f"Saved: {nbr_csv}")

    # ── Plots ────────────────────────────────────────────────────────────

    plots = []
    if not all_recon.empty:
        plots.append(("model_comparison", lambda: plot_model_comparison(all_recon, out_dir, n_layers)))
    if all_residuals:
        plots.append(("interaction_residual", lambda: plot_interaction_residual(all_residuals, out_dir, n_layers)))
    if not all_stability.empty:
        plots.append(("pair_stability", lambda: plot_pair_stability(all_stability, out_dir, n_layers)))
    plots.append(("probe_accuracy", lambda: plot_probe_accuracy(all_probes, out_dir, n_layers)))
    plots.append(("pca_variance", lambda: plot_pca_explained_variance(all_pca, out_dir, n_layers)))
    plots.append(("neighborhood_heatmap", lambda: plot_neighborhood_heatmap(all_nbr, out_dir)))

    pca_layers = [l for l in [0, n_layers // 2, n_layers - 1] if l in all_pca]
    for lid in pca_layers:
        plots.append((f"pca_2d_L{lid}", lambda _l=lid: plot_pca_2d(all_pca[_l]["pca_2d"], prompts, _l, out_dir)))

    for _, fn in tqdm(plots, desc="Plots", unit="plot"):
        fn()

    print(f"\nAll outputs in: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
