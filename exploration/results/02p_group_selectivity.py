"""
02p_group_selectivity.py  (lives in results/)

Auto-discovers groups of semantically related AST classes by clustering them
on their co-activation profiles, then identifies neurons / heads that fire
uniformly and specifically for each group.

METHODOLOGY
-----------
1. Load selectivity arrays  sel[C, L, N]  (z-scored activation contrast per class).
2. Cluster AST classes (rows of mean profile across units) via Ward / correlation.
3. Cut dendrogram at N_GROUPS clusters (swept over several cuts, saved as PDF).
4. For each group G and each unit (l, n):
     group_sel  = mean(sel[G, l, n]) - mean(sel[~G, l, n])
     within_var = std(sel[G, l, n])    # low = all members activate it equally
5. Keep units where group_sel > GROUP_THRESH and within_var < VAR_THRESH.
6. Output:
     - dendrogram with group colouring
     - per-group heatmap (units × classes)
     - summary JSON

OUTPUTS
-------
  output/figures/group_selectivity/{component}_class_dendrogram.pdf
  output/figures/group_selectivity/{component}_group_heatmap_k{k}.pdf   (per k)
  output/figures/group_selectivity/{component}_group_summary_k{k}.json
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
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, leaves_list

# ─────────────────────────────────────────────────────────────────────────────
# Palette (same as other scripts)
# ─────────────────────────────────────────────────────────────────────────────
CLASS_COLOURS = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#999999","#66c2a5","#fc8d62",
    "#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494",
    "#b3b3b3","#1b9e77","#d95f02","#7570b3","#e7298a",
    "#66a61e","#e6ab02","#a6761d","#666666","#8dd3c7",
    "#ffffb3","#bebada","#fb8072","#80b1d3","#fdb462",
    "#b3de69","#fccde5",
]
GROUP_PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9a6324","#fffac8","#800000","#aaffc3",
    "#808000","#ffd8b1","#000075","#a9a9a9",
]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_component(component: str, in_dir: Path, stem: str):
    """Returns sel (C, L, N), mask (L, N), classes list."""
    if component == "heads":
        d      = np.load(in_dir / f"02l_{stem}_head_sel.npz")
        sel    = d["sel"].astype(float)          # (C, L, H)
        mask   = d["ast_pure"].astype(bool)      # (L, H) — keep only AST-pure heads
        # classes in same order as sel (sorted)
        own    = json.load(open(in_dir / f"02l_{stem}_head_ownership.json"))
        classes = sorted(own.keys())
    else:
        npz_key = "neurons" if component == "mlp" else "residual"
        d       = np.load(in_dir / f"02g_{stem}_selectivity_{npz_key}.npz")
        sel           = d["sel"].astype(float)              # (C, L, N)
        core          = d["core"].astype(bool)               # (L, N) — broadly active, not discriminative
        class_neurons = d["class_neurons"].astype(bool)      # (C, L, N)
        # Exclude core neurons first (broadly active, not class-discriminative),
        # then keep only neurons owned by at least one class.
        non_core      = ~core                                 # (L, N)
        mask          = class_neurons.any(axis=0) & non_core # (L, N)
        classes = list(d["classes"])
    return sel, mask, classes

# ─────────────────────────────────────────────────────────────────────────────
# Class clustering
# ─────────────────────────────────────────────────────────────────────────────

def cluster_classes(sel: np.ndarray, mask: np.ndarray):
    """
    Build a (C, M) profile matrix where M = number of active units,
    cluster classes on correlation distance, return linkage Z.
    """
    C, L, N = sel.shape
    # flatten to active units only
    active = mask.reshape(-1)                    # (L*N,)
    flat   = sel.reshape(C, L * N)[:, active]    # (C, M)

    # pairwise correlation distance between classes
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
        # colour each leaf by its group
        leaf_order = leaves_list(Z)
        leaf_colours = [GROUP_PALETTE[(labels_cut[i] - 1) % len(GROUP_PALETTE)]
                        for i in leaf_order]

        def llf(id):
            return classes[id]

        def lc_func(id):
            return leaf_colours[leaf_order.tolist().index(id)] \
                if id < C else "#000000"

        ddata = dendrogram(Z, labels=classes, leaf_rotation=60,
                           leaf_font_size=8, ax=ax,
                           link_color_func=lambda k: "#aaaaaa")

        # colour the leaf labels by group
        for lbl, col in zip(ax.get_xticklabels(),
                             [GROUP_PALETTE[(labels_cut[leaf_order[i]] - 1) % len(GROUP_PALETTE)]
                              for i in range(C)]):
            lbl.set_color(col)

        ax.set_title(f"{component.upper()} — k={k} groups | Ward / correlation distance",
                     fontsize=9)
        ax.set_ylabel("Distance", fontsize=8)

        # legend showing which colour = which group members
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
    """
    For a group defined by class indices group_cidx, score every unit (l, n).
    Returns:
      group_sel  (L, N) — mean(sel[G]) - mean(sel[~G])
      within_var (L, N) — std(sel[G])  low = uniform activation across group
    """
    C, L, N    = sel.shape
    out_cidx   = [c for c in range(C) if c not in group_cidx]
    mu_in      = sel[group_cidx, :, :].mean(axis=0)
    mu_out     = sel[out_cidx,   :, :].mean(axis=0)
    group_sel  = mu_in - mu_out
    within_var = sel[group_cidx, :, :].std(axis=0)
    return group_sel, within_var


def permutation_pvalue(sel, mask, group_cidx, observed_gs, n_perm=500, seed=42):
    """
    Permutation test for group_sel of each active unit.
    Randomly shuffle which classes belong to the "group" (same size as real group)
    n_perm times and compute the null distribution of group_sel.
    Returns p_value (L, N) — proportion of permutations where null >= observed.
    Only computed for active (masked) units; others get p=1.0.
    """
    rng = np.random.default_rng(seed)
    C, L, N = sel.shape
    g_size   = len(group_cidx)
    all_idx  = list(range(C))

    # Flatten to active units only for speed
    flat_mask  = mask.reshape(-1)                         # (L*N,)
    flat_obs   = observed_gs.reshape(-1)[flat_mask]       # (M,)
    flat_sel   = sel.reshape(C, L * N)[:, flat_mask]      # (C, M)

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

GROUP_THRESH = 0.5
VAR_THRESH   = 0.8
ALPHA        = 0.01   # permutation significance level
TOP_K        = 30     # top units shown per group in heatmap
N_PERM       = 5000   # permutation iterations


def plot_group_heatmap(sel, mask, groups, classes, k, component, fmt, dpi, fig_dir):
    """
    For each group at cut k, show a heatmap of top significant units × all classes.
    Units must pass: group_sel > GROUP_THRESH, within_var < VAR_THRESH, p < ALPHA.
    Group member columns highlighted. Stars on y-axis = p < 0.01.
    """
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

        # significant units: threshold + p-value
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

        mat  = np.stack([sel[:, l, n] for l, n in zip(sel_l, sel_n)], axis=0)  # (M, C)
        vmax = np.percentile(np.abs(mat), 99)
        im   = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, interpolation="nearest")

        ax.set_xticks(range(C))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
        for tick_idx, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(g_color if tick_idx in g_cidx else "#333333")
            tick.set_fontweight("bold" if tick_idx in g_cidx else "normal")

        unit_char = 'H' if component == 'heads' else 'N'
        ylabels = [f"{'★' if p < 0.01 else '·'} L{l}{unit_char}{n}  "
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
    """
    Writes one tabular table per k value (no longtable).
    all_summaries: {k: {g_id_str: summary_dict}}
    """
    unit_char = 'H' if component == 'heads' else 'N'
    comp_label = component.replace('_', '')
    lines = []

    for k in sorted(all_summaries.keys()):
        summ   = all_summaries[k]
        label  = f"tab:{comp_label}_groups_k{k}"
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
            # Wrap class list — if long, break across lines with \newline
            cls_list = [r"\texttt{" + c + "}" for c in g["classes"]]
            # Join with commas; let LaTeX wrap naturally in p{} column if needed
            classes_str = ", ".join(cls_list)
            n_sig = g["n_sig"]
            if g["top_units"]:
                tu      = g["top_units"][0]
                top_str = f"L{tu['layer']}{unit_char}{tu['unit']}"
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
# Methodology & findings write-up
# ─────────────────────────────────────────────────────────────────────────────

def write_findings(all_summaries: dict, component: str, classes: list, fig_dir: Path):
    unit_word  = {"heads": "attention head", "mlp": "MLP neuron",
                  "residual": "residual stream dimension"}[component]
    unit_words = unit_word + "s"
    k_str      = ",".join(str(k) for k in sorted(all_summaries.keys()))

    # Stable groups: appear at ≥2 k values with ≥1 sig unit
    group_class_counts: dict[frozenset, list] = {}
    for k, summ in all_summaries.items():
        for g_id_str, g in summ.items():
            if g["n_sig"] > 0:
                key = frozenset(g["classes"])
                group_class_counts.setdefault(key, []).append((k, g["n_sig"]))

    stable = [(cls_set, appearances)
              for cls_set, appearances in group_class_counts.items()
              if len(appearances) >= 2]
    stable.sort(key=lambda x: -sum(n for _, n in x[1]))

    total_sig = {k: sum(g["n_sig"] for g in summ.values())
                 for k, summ in all_summaries.items()}

    component_preamble = {
        "heads": (
            "Attention heads show partial class-level specificity: "
            "34 of the 225 AST-pure heads are monosemantically selective "
            "for a single AST class, while the remaining 191 respond to "
            "two or more classes simultaneously "
            "(Section~\\ref{sec:head_sel}). "
            "This polysemanticity at the individual-class level motivates "
            "a group-level analysis: heads that cannot be assigned uniquely "
            "to one class may nonetheless be specifically tuned to a coherent "
            "syntactic category spanning several related constructs."
        ),
        "mlp": (
            "At the level of individual MLP neurons, class-level uniqueness "
            "is rare. "
            "The per-class analysis identified a single strictly monosemantic "
            "neuron --- MLP neuron~\\#2218 in Layer~5, selective for "
            "\\texttt{Assert} with selectivity index $\\delta = 4.82$ --- "
            "and a small set of class-preferring neurons that do not reach "
            "strict monosemanticity. "
            "This scarcity is consistent with the superposition "
            "hypothesis~\\citep{elhage2022superposition}: MLP neurons "
            "typically encode multiple features in superposition, making "
            "strict per-class uniqueness the exception rather than the rule. "
            "Relaxing the question from individual classes to syntactic "
            "\\emph{groups} reveals a richer landscape of group-selective neurons."
        ),
        "residual": (
            "The residual stream presents the greatest challenge for "
            "per-class uniqueness detection. "
            "As the persistent information highway that carries the hidden state "
            "through every layer --- receiving additive contributions from all "
            "upstream attention heads and MLP sublayers --- residual dimensions "
            "accumulate superpositions of many features. "
            "Consequently, the per-class analysis finds no strictly monosemantic "
            "residual dimensions: every active dimension co-fires for multiple "
            "AST classes. "
            "However, at the group level this picture changes markedly. "
            "Certain residual dimensions activate uniformly and specifically "
            "for syntactically cohesive groups of constructs, revealing that "
            "while the residual stream cannot reliably distinguish individual "
            "node types, it does encode \\emph{abstract syntactic categories} "
            "that generalise across related constructs. "
            "This distinction is practically significant: it suggests that "
            "class-level information is distributed across multiple residual "
            "dimensions, but category-level information is concentrated in "
            "identifiable, group-selective dimensions."
        ),
    }[component]

    # Build findings sentences for stable groups
    findings_sentences = ""
    for cls_set, appearances in stable[:8]:
        cls_list = sorted(cls_set)
        cls_tex  = ", ".join(f"\\texttt{{{c}}}" for c in cls_list)
        max_n    = max(n for _, n in appearances)
        k_vals   = sorted(set(k for k, _ in appearances))
        findings_sentences += (
            f"The syntactic group \\{{{cls_tex}\\}} "
            f"recruits {max_n} group-selective {unit_words} "
            f"(stable across $k = {', '.join(str(k) for k in k_vals)}$). "
        )

    if not findings_sentences:
        findings_sentences = (
            f"No groups yielded significant {unit_words} at $p < {ALPHA}$ "
            f"across multiple values of~$k$, suggesting that at this "
            f"significance threshold the {component} representations do not "
            f"encode group-level syntactic distinctions in a concentrated way. "
        )

    text = (
        f"\\subsection{{Group-Selective {unit_word.title()}s}}\n"
        f"\\label{{sec:group_sel_{component}}}\n\n"
        f"\\paragraph{{Motivation.}}\n"
        f"{component_preamble}\n\n"
        f"\\paragraph{{Methodology.}}\n"
        f"AST classes are clustered hierarchically using Ward linkage on the\n"
        f"correlation distance between their mean selectivity profiles across\n"
        f"all active {unit_words}.\n"
        f"The dendrogram is cut at $k \\in \\{{{k_str}\\}}$ groups.\n"
        f"For each group $\\mathcal{{G}}$ and each active unit $(l, n)$, we compute\n"
        f"the \\emph{{group selectivity score}}\n"
        f"\\begin{{equation}}\n"
        f"  \\Delta\\bar{{s}}_{{\\mathcal{{G}},l,n}} =\n"
        f"    \\frac{{1}}{{|\\mathcal{{G}}|}}\n"
        f"      \\sum_{{c \\in \\mathcal{{G}}}} s_{{c,l,n}}\n"
        f"    -\n"
        f"    \\frac{{1}}{{C - |\\mathcal{{G}}|}}\n"
        f"      \\sum_{{c \\notin \\mathcal{{G}}}} s_{{c,l,n}}\n"
        f"\\end{{equation}}\n"
        f"and the within-group activation standard deviation\n"
        f"$\\sigma_{{\\mathcal{{G}},l,n}} = \\operatorname{{std}}_{{c \\in \\mathcal{{G}}}}(s_{{c,l,n}})$.\n"
        f"A unit is declared group-selective if $\\Delta\\bar{{s}} > {GROUP_THRESH}$\n"
        f"(specificity), $\\sigma_{{\\mathcal{{G}}}} < {VAR_THRESH}$ (uniform activation\n"
        f"across all group members --- this criterion rules out units driven by\n"
        f"a single dominant class within the group and is therefore the key\n"
        f"distinction from the individual-class analysis), and $p < {ALPHA}$\n"
        f"under a permutation test with {N_PERM} random relabellings of classes\n"
        f"into groups of the same size.\n\n"
        f"\\paragraph{{Findings.}}\n"
        f"{findings_sentences}\n"
        f"Across all values of $k$, the total number of significant\n"
        f"group-selective {unit_words} ranges from\n"
        f"{min(total_sig.values())} (at $k={min(all_summaries)}$)\n"
        f"to {max(total_sig.values())} (at $k={max(all_summaries)}$).\n"
        f"The within-group uniformity criterion is critical to the\n"
        f"validity of these findings: without it, a unit that fires\n"
        f"strongly for one member class would be indistinguishable from\n"
        f"a genuinely group-selective unit, conflating individual-class\n"
        f"and category-level specialisation.\n"
        f"Full results are given in Table~\\ref{{tab:{component}_groups}}\n"
        f"and the corresponding heatmaps in\n"
        f"Figure~\\ref{{fig:{component}_group_heatmap}}.\n"
    )

    out = fig_dir / f"{component}_group_writeup.tex"
    out.write_text(text)
    print(f"  saved {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Manual groups — exclusive heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_manual_exclusive_heatmap(sel, mask, manual_groups, classes,
                                  component, fmt, dpi, fig_dir):
    """
    For each manually specified group, find units that are:
      1. Group-selective: gs > GROUP_THRESH, wv < VAR_THRESH, p < ALPHA
      2. EXCLUSIVE: fail (gs > GROUP_THRESH) for every other manual group
    Rows sorted by exclusivity score = gs_target - max(gs_others).
    One subplot per group, side by side.
    """
    C, L, N    = sel.shape
    n_groups   = len(manual_groups)
    unit_char  = 'H' if component == 'heads' else 'N'
    TOP_K      = 40

    # Pre-compute gs, wv, pv for every group
    gs_all, wv_all, pv_all = [], [], []
    for g_name, g_classes in manual_groups:
        g_cidx = [classes.index(c) for c in g_classes if c in classes]
        gs, wv = score_group_selectivity(sel, mask, g_cidx)
        print(f"  [{g_name}]: permutation test …")
        pv     = permutation_pvalue(sel, mask, g_cidx, gs, n_perm=N_PERM)
        gs_all.append(gs); wv_all.append(wv); pv_all.append(pv)

    fig, axes = plt.subplots(1, n_groups,
                             figsize=(5 * n_groups, max(6, C * 0.28)),
                             squeeze=False)
    summary = {}

    for g_idx, (g_name, g_classes) in enumerate(manual_groups):
        ax      = axes[0, g_idx]
        g_cidx  = [classes.index(c) for c in g_classes if c in classes]
        g_color = GROUP_PALETTE[g_idx % len(GROUP_PALETTE)]
        gs, wv, pv = gs_all[g_idx], wv_all[g_idx], pv_all[g_idx]

        # Mask: passes own thresholds
        own_pass = mask & (gs > GROUP_THRESH) & (wv < VAR_THRESH) & (pv < ALPHA)

        # Exclusivity: must FAIL gs > GROUP_THRESH for every other manual group
        for other_idx, (_, _) in enumerate(manual_groups):
            if other_idx == g_idx:
                continue
            own_pass = own_pass & ~(gs_all[other_idx] > GROUP_THRESH)

        active_l, active_n = np.where(own_pass)
        scores = gs[active_l, active_n]
        # Exclusivity score = gs_target - max gs across other groups
        max_other_gs = np.zeros(len(active_l))
        for other_idx, (_, _) in enumerate(manual_groups):
            if other_idx == g_idx:
                continue
            max_other_gs = np.maximum(max_other_gs,
                                      gs_all[other_idx][active_l, active_n])
        excl_scores = scores - max_other_gs
        order    = np.argsort(-excl_scores)[:TOP_K]
        sel_l    = active_l[order]
        sel_n    = active_n[order]
        sc_top   = scores[order]
        ex_top   = excl_scores[order]
        pv_top   = pv[active_l, active_n][order]
        wv_top   = wv[active_l, active_n][order]

        if len(sel_l) == 0:
            ax.text(0.5, 0.5, "No exclusive units", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{g_name}\n(no units)", fontsize=9)
            summary[g_name] = {"classes": g_classes, "n_exclusive": 0, "top_units": []}
            continue

        mat  = np.stack([sel[:, l, n] for l, n in zip(sel_l, sel_n)])  # (M, C)
        vmax = np.percentile(np.abs(mat), 99)
        im   = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, interpolation="nearest")

        ax.set_xticks(range(C))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=6)
        for ti, tick in enumerate(ax.get_xticklabels()):
            tick.set_color(g_color if ti in g_cidx else "#444444")
            tick.set_fontweight("bold" if ti in g_cidx else "normal")

        ylabels = [f"{'★' if p < 0.01 else '·'} L{l}{unit_char}{n} "
                   f"gs={s:.2f} ex={e:.2f} p={p:.3f}"
                   for l, n, s, e, p in
                   zip(sel_l, sel_n, sc_top, ex_top, pv_top)]
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels, fontsize=5.5)

        axb = ax.inset_axes([0, 1.0, 1, 0.04], transform=ax.transAxes)
        axb.imshow([[matplotlib.colors.to_rgba(g_color if ci in g_cidx else "#eeeeee")
                     for ci in range(C)]], aspect="auto", interpolation="nearest")
        axb.set_axis_off()

        n_excl = int(own_pass.sum())
        ax.set_title(f"{g_name}\n{n_excl} exclusive units (top {len(sel_l)} shown)\n"
                     f"sorted by exclusivity = gs − max(gs_others)",
                     fontsize=8, pad=12)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="sel z-score")

        summary[g_name] = {
            "classes":     g_classes,
            "n_exclusive": n_excl,
            "top_units":   [{"layer": int(l), "unit": int(n),
                             "group_sel": float(s), "exclusivity": float(e),
                             "within_var": float(w), "p_value": float(pv[l, n])}
                            for l, n, s, e, w in
                            zip(sel_l, sel_n, sc_top, ex_top, wv_top)],
        }

    fig.suptitle(f"{component.upper()} — exclusive group-selective units "
                 f"(gs>{GROUP_THRESH}, wv<{VAR_THRESH}, p<{ALPHA}, "
                 f"core+class-owned excluded)\n"
                 f"★ p<0.01  ·  p<{ALPHA}  |  ex = exclusivity score",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out = fig_dir / f"{component}_manual_exclusive_heatmap.{fmt}"
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")

    json_out = fig_dir / f"{component}_manual_exclusive_summary.json"
    json.dump(summary, open(json_out, "w"), indent=2)
    print(f"  saved {json_out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stem",      default="contrastive_stubs")
    p.add_argument("--component", default="heads",
                   choices=["heads", "mlp", "residual"])
    p.add_argument("--in_dir",    default="../results/output")
    p.add_argument("--k_list",    default="8,9,10",
                   help="Comma-separated list of k (num groups) to cut at")
    p.add_argument("--format",    default="pdf")
    p.add_argument("--dpi",       type=int, default=200)
    p.add_argument("--manual",    action="store_true",
                   help="Use manually specified groups for residual (exclusive analysis)")
    args = p.parse_args()

    in_dir  = Path(__file__).parent / args.in_dir
    fig_dir = in_dir / "figures" / "group_selectivity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    k_list = [int(k) for k in args.k_list.split(",")]

    print(f"Loading {args.component} selectivity …")
    sel, mask, classes = load_component(args.component, in_dir, args.stem)
    C, L, N = sel.shape
    print(f"  sel shape: {sel.shape}  active units: {mask.sum()}")

    # ── Manual exclusive analysis ────────────────────────────────────────────
    if args.manual:
        manual_groups = [
            ("Yield / YieldFrom",            ["Yield", "YieldFrom"]),
            ("FunctionDef / AsyncFunctionDef",["FunctionDef", "AsyncFunctionDef"]),
            ("ListComp / Subscript",           ["ListComp", "Subscript"]),
            ("DictComp / SetComp",            ["DictComp", "SetComp"]),
        ]
        print("\nRunning manual exclusive group analysis …")
        plot_manual_exclusive_heatmap(sel, mask, manual_groups, classes,
                                      args.component, args.format, args.dpi, fig_dir)
        print("\nDone.")
        return

    print("Clustering AST classes …")
    Z, flat = cluster_classes(sel, mask)

    print("Plotting class dendrogram …")
    plot_class_dendrogram(Z, classes, k_list, args.component,
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
                                  args.component, args.format, args.dpi, fig_dir)
        all_summaries[k] = {str(g_id): v for g_id, v in summ.items()}

    print("\nWriting LaTeX table …")
    write_latex_table(all_summaries, args.component, fig_dir)

    print("Writing findings write-up …")
    write_findings(all_summaries, args.component, classes, fig_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
