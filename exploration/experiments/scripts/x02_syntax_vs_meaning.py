#!/usr/bin/env python3
"""
X02 Phase B — Syntax vs. Meaning (offline analysis)

Reads the intermediate HDF5 produced by x02_extract_prompt_level.py and
determines whether each circuit responds to structural AST patterns (syntax-
stable) or semantic content (meaning-sensitive).

Method:
  - For each (pair, layer), compute the mean activation vector per domain.
  - Jensen-Shannon divergence between domain response distributions:
      Low JSD  → syntax-dominant  (domains respond similarly)
      High JSD → meaning-sensitive (domains diverge)

Usage:
    python experiments/scripts/x02_syntax_vs_meaning.py
    python experiments/scripts/x02_syntax_vs_meaning.py \
        --acts experiments/outputs/x02/prompt_level_activations.h5 \
        --out  experiments/outputs/x02
"""
import argparse
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--acts",
                   default="experiments/outputs/x02/prompt_level_activations.h5")
    p.add_argument("--out", default="experiments/outputs/x02")
    return p.parse_args()


def resolve(path: str) -> Path:
    pp = Path(path)
    return pp if pp.is_absolute() else ROOT / pp


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def jensen_shannon_divergence(vectors: list[np.ndarray]) -> float:
    """
    JSD between K distribution vectors.

    Each vector is treated as an unnormalized distribution over neurons;
    we softmax-normalize before computing JSD.
    """
    if len(vectors) < 2:
        return float("nan")
    distributions = np.array([softmax(np.abs(v).astype(np.float64)) for v in vectors])
    # JSD = H(mean) - mean(H)
    mean_dist = distributions.mean(axis=0)
    eps = 1e-12

    def entropy(p):
        p = p[p > eps]
        return -np.sum(p * np.log(p))

    H_mean = entropy(mean_dist)
    mean_H = np.mean([entropy(d) for d in distributions])
    return float(max(0.0, H_mean - mean_H))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    args = parse_args()
    acts_path = resolve(args.acts)
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not acts_path.exists():
        print(f"ERROR: intermediate file not found: {acts_path}")
        print("Run x02_extract_prompt_level.py first.")
        sys.exit(1)

    print(f"Loading activations: {acts_path}")
    rows = []
    exploded_rows = []
    with h5py.File(str(acts_path), "r") as f:
        meta_attrs = dict(f["metadata"].attrs)
        n_layers = int(meta_attrs.get("n_layers", 8))

        domain_names = [x.decode() if isinstance(x, bytes) else x
                        for x in f["domain_vocab/names"][:]]
        n_domains = len(domain_names)
        print(f"  {n_layers} layers, domains: {domain_names}")

        pairs = list(f["pairs"].keys())
        print(f"  {len(pairs)} pairs")

        for pair_key in pairs:
            pg = f[f"pairs/{pair_key}"]
            domain_ids = pg["prompt_meta/domain_id"][:]  # (N,)

            for lid in range(n_layers):
                layer_key = f"layer_{lid}"
                if layer_key not in pg:
                    continue

                acts_proj = pg[f"{layer_key}/acts_proj_f16"][:].astype(np.float32)  # (N, K)

                # Group by domain — track which did maps to which vec
                domain_vecs = []
                domain_vecs_did = []
                for did in range(n_domains):
                    idx = np.where(domain_ids == did)[0]
                    if len(idx) == 0:
                        continue
                    mean_vec = acts_proj[idx].mean(axis=0)
                    domain_vecs.append(mean_vec)
                    domain_vecs_did.append(did)

                jsd = jensen_shannon_divergence(domain_vecs)

                # Pairwise cosine similarity across domains (syntax stability)
                cos_sims = []
                for i in range(len(domain_vecs)):
                    for j in range(i + 1, len(domain_vecs)):
                        cos_sims.append(cosine_similarity(domain_vecs[i], domain_vecs[j]))
                mean_cos = float(np.mean(cos_sims)) if cos_sims else float("nan")

                # Residual stream baseline cosine similarity
                resid_cos = float("nan")
                resid_domain_vecs = []
                resid_domain_vecs_did = []
                if "resid_f16" in pg[layer_key]:
                    resid = pg[f"{layer_key}/resid_f16"][:].astype(np.float32)  # (N, d_model)
                    for did in range(n_domains):
                        idx = np.where(domain_ids == did)[0]
                        if len(idx) == 0:
                            continue
                        resid_domain_vecs.append(resid[idx].mean(axis=0))
                        resid_domain_vecs_did.append(did)
                    resid_cos_sims = []
                    for i in range(len(resid_domain_vecs)):
                        for j in range(i + 1, len(resid_domain_vecs)):
                            resid_cos_sims.append(
                                cosine_similarity(resid_domain_vecs[i], resid_domain_vecs[j]))
                    resid_cos = float(np.mean(resid_cos_sims)) if resid_cos_sims else float("nan")

                rows.append({
                    "pair": pair_key,
                    "layer": lid,
                    "jsd": jsd,
                    "mean_cross_domain_cosine": mean_cos,
                    "resid_cross_domain_cosine": resid_cos,
                    "n_domains_present": len(domain_vecs),
                    "n_neurons_in_circuit": acts_proj.shape[1],
                })

                # Exploded per-domain-pair rows
                for i in range(len(domain_vecs)):
                    for j in range(i + 1, len(domain_vecs)):
                        di, dj = domain_vecs_did[i], domain_vecs_did[j]
                        dp_label = f"{domain_names[di]}\u2194{domain_names[dj]}"
                        circuit_cos = cosine_similarity(domain_vecs[i], domain_vecs[j])
                        # Residual cosine for this specific domain pair
                        r_cos = float("nan")
                        if resid_domain_vecs:
                            ri = resid_domain_vecs_did.index(di) if di in resid_domain_vecs_did else None
                            rj = resid_domain_vecs_did.index(dj) if dj in resid_domain_vecs_did else None
                            if ri is not None and rj is not None:
                                r_cos = cosine_similarity(resid_domain_vecs[ri], resid_domain_vecs[rj])
                        exploded_rows.append({
                            "pair": pair_key,
                            "layer": lid,
                            "jsd": jsd,
                            "domain_pair": dp_label,
                            "circuit_cosine": circuit_cos,
                            "resid_cosine": r_cos,
                            "n_neurons_in_circuit": acts_proj.shape[1],
                        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data found — check that activation HDF5 is non-empty.")
        return

    # ── Ranked circuits CSV ────────────────────────────────────────────────
    ranked = df.sort_values("jsd").copy()
    ranked["interpretation"] = ranked["jsd"].apply(
        lambda j: "syntax-dominant" if j < ranked["jsd"].median() else "meaning-sensitive"
    )
    csv_path = out_dir / "ranked_circuits.csv"
    ranked.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(ranked.head(20).to_string(index=False))

    # ── Exploded per-domain-pair CSV ──────────────────────────────────────
    if exploded_rows:
        exploded_df = pd.DataFrame(exploded_rows)
        exploded_df["interpretation"] = exploded_df["jsd"].apply(
            lambda j: "syntax-dominant" if j < df["jsd"].median() else "meaning-sensitive"
        )
        exploded_csv = out_dir / "ranked_circuits_by_domain_pair.csv"
        exploded_df.to_csv(exploded_csv, index=False)
        print(f"\nSaved: {exploded_csv}  ({len(exploded_df)} rows, "
              f"{len(exploded_df) / len(df):.1f}x aggregated)")

    # ── JSD heatmap: pairs × layers ────────────────────────────────────────
    pairs_list = sorted(df["pair"].unique())
    matrix = np.full((len(pairs_list), n_layers), fill_value=np.nan)
    for row in df.itertuples():
        pi = pairs_list.index(row.pair)
        matrix[pi, row.layer] = row.jsd

    n_pairs = len(pairs_list)
    fig_h = max(12, n_pairs * 0.18)   # scale height with pair count
    fig_w = max(10, n_layers * 1.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0)
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([f"L{i}" for i in range(n_layers)])
    # For very large pair counts, only label every Nth pair
    if n_pairs > 80:
        tick_step = max(1, n_pairs // 80)
        tick_positions = list(range(0, n_pairs, tick_step))
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([pairs_list[i] for i in tick_positions], fontsize=4)
    else:
        ax.set_yticks(range(n_pairs))
        ax.set_yticklabels(pairs_list, fontsize=5)
    fig.colorbar(im, ax=ax, label="JSD (high = meaning-sensitive)", shrink=0.6)
    ax.set_title("Domain JSD per circuit × layer\n"
                 "(green=syntax-stable, red=meaning-sensitive)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Circuit (pair)")
    fig.tight_layout()
    heatmap_path = out_dir / "domain_jsd_heatmap.png"
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
