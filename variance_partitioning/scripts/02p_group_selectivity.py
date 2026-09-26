"""
02p_group_selectivity.py  (submission version)

Auto-discovers groups of semantically related AST classes by Ward clustering
on their co-activation profiles, then identifies residual-stream dimensions
that fire uniformly and specifically for each group.

OUTPUTS  (under out_dir/figures/group_selectivity/)
  residual_class_dendrogram.pdf
  residual_group_heatmap_k{k}.pdf      one per k in --k_list
  residual_group_summary_k{k}.json     one per k in --k_list
  residual_group_table.tex             LaTeX summary table

Usage
-----
  python 02p_group_selectivity.py --stem contrastive_stubs \\
         --in_dir ../results/output --k_list 14
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

GROUP_PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9a6324","#fffac8","#800000","#aaffc3",
    "#808000","#ffd8b1","#000075","#a9a9a9",
]

GROUP_THRESH = 0.5
VAR_THRESH   = 0.8
ALPHA        = 0.01
TOP_K        = 30
N_PERM       = 5000


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — residual only
# ─────────────────────────────────────────────────────────────────────────────

def load_residual(in_dir: Path, stem: str):
    """Returns sel (C, L, N), mask (L, N), classes list."""
    d             = np.load(in_dir / f"02g_{stem}_selectivity_residual.npz")
    sel           = d["sel"].astype(float)              # (C, L, N)
    core          = d["core"].astype(bool)              # (L, N)
    class_neurons = d["class_neurons"].astype(bool)     # (C, L, N)
    non_core      = ~core
    mask          = class_neurons.any(axis=0) & non_core
    classes       = list(d["classes"])
    return sel, mask, classes


# ─────────────────────────────────────────────────────────────────────────────
# Class clustering
# ─────────────────────────────────────────────────────────────────────────────

def cluster_classes(sel: np.ndarray, mask: np.ndarray):
    """Ward linkage on correlation distance between class profiles."""
    C, L, N = sel.shape
    active = mask.reshape(-1)
    flat   = sel.reshape(C, L * N)[:, active]    # (C, M)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist = pdist(flat, metric="correlation")
        dist = np.nan_to_num(dist, nan=1.0)
    Z = linkage(dist, method="ward")
    return Z, flat


# ─────────────────────────────────────────────────────────────────────────────
# Dendrogram figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_dendrogram(Z, classes, k_list, component, fmt, dpi, fig_dir):
    C = len(classes)
    fig, axes = plt.subplots(len(k_list), 1,
                             figsize=(max(10, C * 0.45), 4 * len(k_list)))
    if len(k_list) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_list):
        labels_cut = fcluster(Z, k, criterion="maxclust")
        leaf_order = leaves_list(Z)

        ddata = dendrogram(Z, labels=classes, leaf_rotation=60,
                           leaf_font_size=8, ax=ax,
                           link_color_func=lambda k: "#aaaaaa")

        for lbl, col in zip(ax.get_xticklabels(),
                             [GROUP_PALETTE[(labels_cut[leaf_order[i]] - 1) % len(GROUP_PALETTE)]
                              for i in range(C)]):
            lbl.set_color(col)

        ax.set_title(f"{component.upper()} — k={k} groups | Ward / correlation distance",
                     fontsize=9)
        ax.set_ylabel("Distance", fontsize=8)

        group_labels = {}
        for i, cls in enumerate(classes):
            g = labels_cut[i]
            group_labels.setdefault(g, []).append(cls)
        handles = [mpatches.Patch(
                       facecolor=GROUP_PALETTE[(g - 1) % len(GROUP_PALETTE)],
                       label=f"G{g}: " + ", ".join(group_labels[g]))
                   for g in sorted(group_labels)]
        ax.legend(handles=handles, fontsize=6.5, loc="upper right",
                  frameon=True, ncol=1)

    fig.tight_layout()
    out = fig_dir / f"{component}_class_dendrogram.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Group selectivity scoring + permutation significance
# ─────────────────────────────────────────────────────────────────────────────

def score_group_selectivity(sel, mask, group_cidx):
    C, L, N   = sel.shape
    out_cidx  = [c for c in range(C) if c not in group_cidx]
    mu_in     = sel[group_cidx, :, :].mean(axis=0)
    mu_out    = sel[out_cidx,   :, :].mean(axis=0)
    group_sel  = mu_in - mu_out
    within_var = sel[group_cidx, :, :].std(axis=0)
    return group_sel, within_var


def permutation_pvalue(sel, mask, group_cidx, observed_gs, n_perm=N_PERM, seed=42):
    rng = np.random.default_rng(seed)
    C, L, N  = sel.shape
    g_size   = len(group_cidx)
    all_idx  = list(range(C))

    flat_mask = mask.reshape(-1)
    flat_obs  = observed_gs.reshape(-1)[flat_mask]
    flat_sel  = sel.reshape(C, L * N)[:, flat_mask]

    null_ge_obs = np.zeros(flat_obs.shape[0], dtype=int)
    for _ in range(n_perm):
        perm_g   = rng.choice(all_idx, size=g_size, replace=False)
        perm_out = [c for c in all_idx if c not in perm_g]
        null_gs  = flat_sel[perm_g, :].mean(axis=0) - flat_sel[perm_out, :].mean(axis=0)
        null_ge_obs += (null_gs >= flat_obs).astype(int)

    p_flat = null_ge_obs / n_perm
    p_full = np.ones((L * N,))
    p_full[flat_mask] = p_flat
    return p_full.reshape(L, N)


# ─────────────────────────────────────────────────────────────────────────────
# Per-group heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_group_heatmap(sel, mask, groups, classes, k, component, fmt, dpi, fig_dir):
    C, L, N  = sel.shape
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups,
                             figsize=(5 * n_groups, max(5, C * 0.28)),
                             squeeze=False)
    summary = {}

    for g_idx, (g_id, g_cidx, g_name) in enumerate(groups):
        ax      = axes[0, g_idx]
        g_color = GROUP_PALETTE[(g_id - 1) % len(GROUP_PALETTE)]

        gs, wv = score_group_selectivity(sel, mask, g_cidx)

        print(f"    G{g_id}: permutation test ({N_PERM} perms) …")
        pv = permutation_pvalue(sel, mask, g_cidx, gs, n_perm=N_PERM)

        sig = mask & (gs > GROUP_THRESH) & (wv < VAR_THRESH) & (pv < ALPHA)
        active_l, active_n = np.where(sig)
        scores  = gs[active_l, active_n]
        pvals   = pv[active_l, active_n]
        wvs     = wv[active_l, active_n]
        order   = np.argsort(-scores)[:TOP_K]
        sel_l   = active_l[order]
        sel_n   = active_n[order]
        scores_top = scores[order]
        pvals_top  = pvals[order]
        wvs_top    = wvs[order]

        if len(sel_l) == 0:
            ax.text(0.5, 0.5, "No significant units\npass thresholds",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"G{g_id}: {g_name}\n(no units)", fontsize=8)
            summary[g_id] = {"classes": [classes[c] for c in g_cidx], "n_sig": 0,
                             "top_units": []}
            continue

        mat  = np.stack([sel[:, l, n] for l, n in zip(sel_l, sel_n)], axis=0)
        vmax = np.percentile(np.abs(mat), 99)
        im   = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, interpolation="nearest")

        ax.set_xticks(range(C))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
        for tick_idx, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(g_color if tick_idx in g_cidx else "#333333")
            tick.set_fontweight("bold" if tick_idx in g_cidx else "normal")

        ylabels = [f"{'★' if p < 0.01 else '·'} L{l}N{n}  "
                   f"gs={s:.2f} wv={w:.2f} p={p:.3f}"
                   for l, n, s, w, p in
                   zip(sel_l, sel_n, scores_top, wvs_top, pvals_top)]
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=5.5)

        axb = ax.inset_axes([0, 1.0, 1, 0.04], transform=ax.transAxes)
        axb.imshow([[matplotlib.colors.to_rgba(g_color if c in g_cidx else "#eeeeee")
                     for c in range(C)]], aspect="auto", interpolation="nearest")
        axb.set_axis_off()

        n_total_sig = int(sig.sum())
        ax.set_title(f"G{g_id}: {g_name}\n"
                     f"gs>{GROUP_THRESH}, wv<{VAR_THRESH}, p<{ALPHA}  "
                     f"| {n_total_sig} sig. units (top {len(sel_l)} shown)",
                     fontsize=8, pad=12)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="sel z-score")

        summary[g_id] = {
            "classes":   [classes[c] for c in g_cidx],
            "n_sig":     n_total_sig,
            "top_units": [{"layer": int(l), "unit": int(n),
                           "group_sel": float(s), "within_var": float(w),
                           "p_value":   float(p)}
                          for l, n, s, w, p in
                          zip(sel_l, sel_n, scores_top, wvs_top, pvals_top)],
        }

    fig.suptitle(f"{component.upper()} group selectivity  k={k}  "
                 f"(★ = p<0.01,  · = p<{ALPHA})",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = fig_dir / f"{component}_group_heatmap_k{k}.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

    summary_out = fig_dir / f"{component}_group_summary_k{k}.json"
    json.dump({str(gid): v for gid, v in summary.items()},
              open(summary_out, "w"), indent=2)
    print(f"  saved {summary_out}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX summary table
# ─────────────────────────────────────────────────────────────────────────────

def write_latex_table(all_summaries: dict, component: str, fig_dir: Path):
    """Writes one tabular table per k value."""
    lines = []
    for k in sorted(all_summaries.keys()):
        summ   = all_summaries[k]
        label  = f"tab:{component}_groups_k{k}"
        caption = (f"Group-selective {component.replace('_',' ')} units at $k={k}$ groups "
                   f"($\\Delta\\bar{{s}} > {GROUP_THRESH}$, "
                   f"$\\sigma_{{\\mathcal{{G}}}} < {VAR_THRESH}$, $p < {ALPHA}$, "
                   f"core-class units excluded). "
                   f"$\\star$ denotes top unit; \\#sig = total significant units in group.")

        lines += [
            f"\\begin{{table}}[htbp]",
            f"\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"{\small",
            r"\begin{tabular}{llrllll}",
            r"\toprule",
            r"Group & AST classes & \#sig & Top unit & $\Delta\bar{s}$ & $\sigma_{\mathcal{G}}$ & $p$ \\",
            r"\midrule",
        ]

        for g_id_str in sorted(summ.keys(), key=lambda x: int(x)):
            g = summ[g_id_str]
            cls_list = [r"\texttt{" + c + "}" for c in g["classes"]]
            classes_str = ", ".join(cls_list)
            n_sig = g["n_sig"]
            if g["top_units"]:
                tu      = g["top_units"][0]
                top_str = f"L{tu['layer']}N{tu['unit']}"
                gs_str  = f"{tu['group_sel']:.3f}"
                wv_str  = f"{tu['within_var']:.3f}"
                p_val   = tu['p_value']
                p_str   = f"{p_val:.3f}" if p_val >= 0.001 else "$<$0.001"
            else:
                top_str = gs_str = wv_str = p_str = "---"
            lines.append(
                f"G{g_id_str} & {classes_str} & {n_sig} & "
                f"\\texttt{{{top_str}}} & {gs_str} & {wv_str} & {p_str} \\\\"
            )

        lines += [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]

    out = fig_dir / f"{component}_group_table.tex"
    out.write_text("\n".join(lines))
    print(f"  saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem",      default="contrastive_stubs")
    p.add_argument("--in_dir",    default="../results/output")
    p.add_argument("--k_list",    default="14",
                   help="Comma-separated list of k (num groups) to cut at")
    p.add_argument("--format",    default="pdf")
    p.add_argument("--dpi",       type=int, default=200)
    args = p.parse_args()

    component = "residual"
    in_dir    = Path(args.in_dir)
    fig_dir   = in_dir / "figures" / "group_selectivity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    k_list = [int(k) for k in args.k_list.split(",")]

    print(f"Loading {component} selectivity …")
    sel, mask, classes = load_residual(in_dir, args.stem)
    C, L, N = sel.shape
    print(f"  sel shape: {sel.shape}  active units: {mask.sum()}")

    print("Clustering AST classes …")
    Z, flat = cluster_classes(sel, mask)

    print("Plotting class dendrogram …")
    plot_class_dendrogram(Z, classes, k_list, component,
                          args.format, args.dpi, fig_dir)

    all_summaries = {}
    for k in k_list:
        print(f"\nScoring group selectivity  k={k} …")
        labels = fcluster(Z, k, criterion="maxclust")
        groups = []
        for g_id in sorted(set(labels)):
            g_cidx = [i for i, l in enumerate(labels) if l == g_id]
            g_name = " / ".join(classes[i] for i in g_cidx)
            if len(g_name) > 40:
                g_name = " / ".join(classes[i] for i in g_cidx[:3]) + " …"
            groups.append((g_id, g_cidx, g_name))
            print(f"  G{g_id}: {[classes[i] for i in g_cidx]}")

        summ = plot_group_heatmap(sel, mask, groups, classes, k,
                                  component, args.format, args.dpi, fig_dir)
        all_summaries[k] = {str(g_id): v for g_id, v in summ.items()}

    print("\nWriting LaTeX table …")
    write_latex_table(all_summaries, component, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
