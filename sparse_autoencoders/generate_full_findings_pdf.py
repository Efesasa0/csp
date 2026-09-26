import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from core import data_path  # noqa: E402
#!/usr/bin/env python3
"""Comprehensive PDF report enumerating every SAE finding from the ATLAS pipeline.

For each finding the report contains a title card, a plain-language explanation,
the supporting data tables, and the supporting plots (regenerated from cached
JSON — not just thumbnails of pre-saved PNGs).

Output: data/reports/SAE_All_Findings_Report.pdf
"""

import json
from collections import defaultdict
from pathlib import Path
from textwrap import fill

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

# ─────────────────────────── paths & constants ────────────────────────────
BASE    = Path(data_path())
CACHE   = BASE / "cache"
ADV     = CACHE / "advanced"
ADV2    = CACHE / "advanced2"
RESULTS = BASE / "results"
REPORTS = BASE / "reports"
SWEEP   = BASE / "sweep_results"

OUT_PDF = REPORTS / "SAE_All_Findings_Report.pdf"

NODES = ["For", "While", "If", "FunctionDef", "AsyncFunctionDef", "ClassDef",
         "With", "Try", "Lambda", "Return", "Yield", "ListComp", "Assert"]
SITES_4 = [(4, "mlp"), (4, "resid"), (5, "mlp"), (5, "resid"),
           (6, "mlp"), (6, "resid"), (7, "mlp"), (7, "resid")]
LAYERS_4 = [4, 5, 6, 7]
SITES = ["mlp", "resid"]

MLP_COL   = "#e74c3c"
RESID_COL = "#3498db"
BG_COL    = "#f8fafc"
GRID_COL  = "#e2e8f0"
DARK      = "#1e293b"


# ─────────────────────────── load helpers ────────────────────────────
def load(path):
    with open(path) as f:
        return json.load(f)


def try_load(path):
    path = Path(path)
    return load(path) if path.exists() else None


# ─────────────────────────── layout primitives ────────────────────────────
def cover_page(pdf):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor(DARK)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.68, "SAE Findings — Full Report",
            ha="center", va="center", fontsize=30, fontweight="bold",
            color="white")
    ax.text(0.5, 0.56,
            "Every interpretability finding from the ATLAS pipeline,\n"
            "with supporting plots, data, and explanation.",
            ha="center", va="center", fontsize=13, color="#cbd5e1",
            style="italic")
    ax.text(0.5, 0.38,
            "Model: Pythia-70M · Sites: L4–L7 (MLP-out and residual)\n"
            "AST classes: For, While, If, FunctionDef, AsyncFunctionDef,\n"
            "ClassDef, With, Try, Lambda, Return, Yield, ListComp, Assert",
            ha="center", va="center", fontsize=10, color="#94a3b8",
            fontfamily="monospace")
    ax.text(0.5, 0.18, "COMP0087 Coursework",
            ha="center", va="center", fontsize=11, color="#64748b")
    pdf.savefig(fig); plt.close(fig)


def finding_title(pdf, number, title, subtitle):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COL)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.62, f"Finding {number}", ha="center", va="center",
            fontsize=18, color="#64748b", fontweight="bold")
    ax.text(0.5, 0.50, title, ha="center", va="center",
            fontsize=26, fontweight="bold", color=DARK)
    ax.text(0.5, 0.38, fill(subtitle, 70),
            ha="center", va="center", fontsize=13, color="#475569",
            style="italic")
    # divider
    ax.plot([0.2, 0.8], [0.30, 0.30], color="#94a3b8", lw=1)
    pdf.savefig(fig); plt.close(fig)


def explanation_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COL)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=16,
            fontweight="bold", color=DARK, va="top")

    y = 0.87
    for p in paragraphs:
        wrapped = fill(p, 95)
        ax.text(0.05, y, wrapped, transform=ax.transAxes, fontsize=11,
                color="#1f2937", va="top", linespacing=1.45)
        y -= 0.045 * (wrapped.count("\n") + 2) + 0.01
    pdf.savefig(fig); plt.close(fig)


def table_page(pdf, title, col_labels, rows, note=None, fontsize=8):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COL)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.96, title, ha="center", va="top",
            fontsize=14, fontweight="bold", color=DARK,
            transform=ax.transAxes)

    if rows:
        table = ax.table(cellText=rows, colLabels=col_labels,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.scale(1, 1.25)
        for (r, c), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#334155")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#f1f5f9" if r % 2 == 0 else "white")
    else:
        ax.text(0.5, 0.5, "(no data available)", ha="center", va="center",
                fontsize=12, color="#94a3b8")

    if note:
        ax.text(0.5, 0.04, note, ha="center", va="bottom",
                fontsize=8, color="#64748b", style="italic",
                transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


def save_fig(pdf, fig):
    fig.tight_layout()
    pdf.savefig(fig); plt.close(fig)


# =========================================================================
#  FINDING 1 — SAE Reconstruction Quality
# =========================================================================
def finding_1_r2(pdf):
    retrain = load(RESULTS / "retrain_summary.json")
    mlp_r2, resid_r2 = {}, {}
    for v in retrain.values():
        (mlp_r2 if v["site"] == "mlp" else resid_r2)[v["layer"]] = v["r2"]

    finding_title(pdf, 1, "SAE Reconstruction Quality",
                  "How well do our trained SAEs reconstruct the underlying activations?")

    mlp_vals = [mlp_r2[l] for l in sorted(mlp_r2)]
    res_vals = [resid_r2[l] for l in sorted(resid_r2)]
    mlp_better = sum(1 for l in range(8) if mlp_r2.get(l, 0) > resid_r2.get(l, 0))

    explanation_page(pdf, "Finding 1 — Explanation", [
        "We trained a 32-latent top-k SAE at every layer and site (MLP output and residual "
        "stream, layers 0–7). After training, we measured the R² of the reconstruction on a "
        "held-out set of activations — i.e. how much of the variance the sparse reconstruction "
        "can explain. This is the usual sanity check: if R² is low, no downstream analysis is "
        "trustworthy because the SAE is just lossy noise.",

        f"Both MLP and residual SAEs sit comfortably above R² = 0.9 at all layers. The mean is "
        f"{np.mean(mlp_vals):.3f} for MLP (range {min(mlp_vals):.3f}–{max(mlp_vals):.3f}) and "
        f"{np.mean(res_vals):.3f} for residual (range {min(res_vals):.3f}–{max(res_vals):.3f}). "
        f"MLP SAEs have the higher R² in {mlp_better}/8 layers — MLP activations are evidently "
        f"easier to reconstruct with a sparse code.",

        "This is the first hint of an important dissociation that reappears in Finding 3: the "
        "SAE with the higher R² (MLP) turns out to be the less faithful one when we actually "
        "probe its latents for syntax. High reconstruction R² is necessary but not sufficient "
        "for interpretability."
    ])

    # Table
    rows = []
    for l in range(8):
        m = mlp_r2.get(l, float("nan"))
        r = resid_r2.get(l, float("nan"))
        rows.append([f"L{l}", f"{m:.4f}", f"{r:.4f}", f"{m-r:+.4f}"])
    table_page(pdf, "R² by layer and site",
               ["Layer", "MLP R²", "Residual R²", "Δ (MLP − Resid)"], rows)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_COL)
    layers = list(range(8))
    ax.plot(layers, mlp_vals, "o-", label="MLP", color=MLP_COL, lw=2, ms=8)
    ax.plot(layers, res_vals, "s-", label="Residual", color=RESID_COL, lw=2, ms=8)
    ax.set_xlabel("Layer"); ax.set_ylabel("Held-out R²")
    ax.set_title("SAE reconstruction quality by layer", fontweight="bold")
    ax.set_xticks(layers)
    ax.axhline(0.9, ls="--", color="gray", alpha=0.5, label="R² = 0.9")
    ax.set_ylim(0.7, 1.0)
    ax.legend(); ax.grid(color=GRID_COL)
    ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 2 — Feature count phase transition
# =========================================================================
def finding_2_feature_counts(pdf):
    feat_counts = load(RESULTS / "multilayer_feature_counts.json")
    layer_totals = {int(l): sum(feat_counts[l].values()) for l in feat_counts}
    early = sum(layer_totals[l] for l in range(4))
    late  = sum(layer_totals[l] for l in range(4, 8))

    finding_title(pdf, 2, "Phase Transition in Feature Count",
                  "Do selective AST features emerge all at once, or gradually across layers?")

    explanation_page(pdf, "Finding 2 — Explanation", [
        "At every layer×site we count features whose activation is selective for a specific AST "
        "node — i.e. the feature fires on, say, For loops much more than on everything else. "
        "This is the pool of interpretable features the rest of the pipeline studies.",

        f"The count per layer is not uniform: layers 0–3 host only {early} selective features in "
        f"total across all 13 AST classes, while layers 4–7 host {late} "
        f"({late/max(early,1):.1f}× more). This looks like a phase transition — the model "
        "evidently does not form discrete syntactic representations until roughly halfway "
        "through the network.",

        "The effect is distributed across AST classes, not a handful of outliers. It matches the "
        "standard picture of transformer layer hierarchies: early layers deal in tokens, late "
        "layers in structured, abstract concepts — and 'AST node type' lives in the latter camp. "
        "This is why the rest of the analyses focus on L4–L7 only."
    ])

    # Per-node per-layer table
    rows = []
    for layer_str in sorted(feat_counts.keys(), key=int):
        row = [f"L{layer_str}"] + [str(feat_counts[layer_str].get(n, 0)) for n in NODES]
        row.append(str(sum(feat_counts[layer_str].values())))
        rows.append(row)
    table_page(pdf, "Selective features per (layer × node)",
               ["Layer"] + [n[:5] for n in NODES] + ["TOTAL"], rows, fontsize=7)

    # Stacked bar
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(BG_COL)
    layers = list(range(8))
    bottom = np.zeros(8)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(NODES)))
    for i, n in enumerate(NODES):
        vals = [feat_counts[str(l)].get(n, 0) for l in layers]
        ax.bar(layers, vals, bottom=bottom, label=n, color=cmap[i])
        bottom += np.array(vals)
    ax.axvspan(-0.5, 3.5, alpha=0.08, color="gray", label="early (few features)")
    ax.set_xlabel("Layer"); ax.set_ylabel("Selective features (stacked by AST node)")
    ax.set_title("Phase transition in feature count", fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.set_facecolor(BG_COL); ax.grid(axis="y", color=GRID_COL)
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 3 — Linear probe: Raw vs SAE latents
# =========================================================================
def finding_3_linear_probe(pdf):
    probe_data = {f"L{l}_{s}": load(ADV2 / f"linear_probe_L{l}_{s}.json")
                  for l, s in SITES_4}

    finding_title(pdf, 3, "Linear Probe: Raw vs SAE",
                  "Do SAE latents preserve the linearly-decodable syntax information "
                  "that lives in raw activations?")

    resid_drops = []
    mlp_drops = []
    for key, d in probe_data.items():
        drop = d["raw_activation_probe"]["accuracy"] - d["sae_latent_probe"]["accuracy"]
        (resid_drops if "resid" in key else mlp_drops).append(drop)

    explanation_page(pdf, "Finding 3 — Explanation", [
        "We train two linear probes on the same classification task (AST node from activation): "
        "one on raw activations, one on the SAE-encoded latents. The gap between them is the "
        "'faithfulness penalty' imposed by the sparse decomposition — information that exists in "
        "the raw vector but is destroyed by compressing it through the SAE.",

        f"Residual SAEs lose only {np.mean(resid_drops)*100:+.2f} pp of probe accuracy on "
        f"average. MLP SAEs lose {np.mean(mlp_drops)*100:+.2f} pp — roughly "
        f"{np.mean(mlp_drops)/max(np.mean(resid_drops),1e-6):.1f}× as much.",

        "Read together with Finding 1, this is the key dissociation: the MLP SAE reconstructs "
        "better (higher R²), but the residual SAE is more faithful for interpretability "
        "purposes. High R² does not imply high probe preservation. This is the kind of thing "
        "that should make anyone quoting 'our SAE has R² = 0.96' a bit more careful.",

        "The per-class breakdown on the strongest site (L7 residual) also reveals a three-tier "
        "structure on the classes themselves — 'easy' classes (While, Lambda, Assert, With, "
        "AsyncFunctionDef ≈ 100%), 'moderate' (ClassDef, If, Try, Yield ≈ 60–90%), and 'hard' "
        "(FunctionDef, Return, For, ListComp ≲ 50%). The hard tier looks structural: "
        "FunctionDef overlaps with ClassDef bodies, Return with FunctionDef, For with ListComp."
    ])

    # Overall table
    rows = []
    for key, d in probe_data.items():
        raw_a = d["raw_activation_probe"]["accuracy"]
        sae_a = d["sae_latent_probe"]["accuracy"]
        raw_f = d["raw_activation_probe"]["weighted_f1"]
        sae_f = d["sae_latent_probe"]["weighted_f1"]
        rows.append([key,
                     f"{raw_a:.4f}", f"{sae_a:.4f}", f"{raw_a-sae_a:+.4f}",
                     f"{raw_f:.4f}", f"{sae_f:.4f}", f"{raw_f-sae_f:+.4f}"])
    table_page(pdf, "Linear probe: Raw vs SAE accuracy and weighted F1",
               ["Site", "Raw Acc", "SAE Acc", "Acc Drop",
                "Raw F1", "SAE F1", "F1 Drop"], rows)

    # Per-class for L7_resid
    best = probe_data["L7_resid"]
    raw_cls = best["raw_activation_probe"]["per_class_accuracy"]
    sae_cls = best["sae_latent_probe"]["per_class_accuracy"]
    rows = []
    for n in NODES:
        r, s = raw_cls[n], sae_cls[n]
        verdict = ("EASY" if r >= 0.9 else "MODERATE" if r >= 0.6
                   else "HARD" if r >= 0.3 else "FAILING")
        rows.append([n, f"{r:.3f}", f"{s:.3f}", f"{r-s:+.3f}", verdict])
    table_page(pdf, "Per-class probe accuracy at L7 residual (best site)",
               ["AST node", "Raw", "SAE", "Drop", "Verdict"], rows)

    # Plot 1 — overall comparison
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG_COL)
    sites = list(probe_data.keys())
    raw_accs = [probe_data[s]["raw_activation_probe"]["accuracy"] for s in sites]
    sae_accs = [probe_data[s]["sae_latent_probe"]["accuracy"] for s in sites]
    x = np.arange(len(sites)); w = 0.35
    ax.bar(x - w/2, raw_accs, w, label="Raw activation", color="#6366f1")
    ax.bar(x + w/2, sae_accs, w, label="SAE latent", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels(sites, rotation=15)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0.3, 0.85)
    ax.set_title("Linear probe accuracy — raw vs SAE", fontweight="bold")
    ax.legend(); ax.grid(axis="y", color=GRID_COL); ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)

    # Plot 2 — per-class L7 resid
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG_COL)
    raw_vals = [raw_cls[n] for n in NODES]
    sae_vals = [sae_cls[n] for n in NODES]
    x = np.arange(len(NODES))
    ax.bar(x - w/2, raw_vals, w, label="Raw", color="#6366f1")
    ax.bar(x + w/2, sae_vals, w, label="SAE", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels(NODES, rotation=40, ha="right")
    ax.set_ylabel("Per-class accuracy"); ax.set_ylim(0, 1.05)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
    ax.set_title("Per-class probe accuracy at L7 residual", fontweight="bold")
    ax.legend(); ax.grid(axis="y", color=GRID_COL); ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 4 — Causal ablation
# =========================================================================
def finding_4_causal_ablation(pdf):
    ablation = {f"L{l}_{s}": load(ADV2 / f"causal_ablation_L{l}_{s}.json")
                for l, s in SITES_4}

    finding_title(pdf, 4, "Causal Ablation",
                  "When we zero a single selective feature, does the model's "
                  "probability of the matching keyword actually drop?")

    explanation_page(pdf, "Finding 4 — Explanation", [
        "For each (layer, site, AST node) we take the top selective feature, ablate it (zero its "
        "activation) during a forward pass on 100 prompts, and measure the drop in the model's "
        "probability of producing the AST node's keyword. 'relative_drop' is "
        "(baseline − ablated) / baseline; positive values mean ablation hurts (expected).",

        "Effects are generally small in absolute terms (baselines themselves are tiny — keywords "
        "sit in the 10⁻⁴ probability range), with two consequences. First: syntax appears to be "
        "encoded redundantly across many features, not concentrated in single ones. Ablating one "
        "is a tap on the shoulder, not a switch. Second: L4 residual shows anomalously large "
        "relative drops for a few nodes (AsyncFunctionDef, ListComp) — the only place in the "
        "network where single-feature ablation has a clean causal signature.",

        "A nice consistency check: the per-node ablation effect at L7 resid has only weak "
        "correlation with the probe accuracy of the same node at the same site. A feature can "
        "be linearly separable without being causally load-bearing, and vice versa. The two "
        "measures capture genuinely different aspects of 'feature importance'."
    ])

    # Summary table
    rows = []
    for key, d in ablation.items():
        positive = sum(1 for n in NODES if d[n]["prob_drop"] > 0)
        mean_abs = np.mean([abs(d[n]["relative_drop"]) for n in NODES])
        max_node = max(NODES, key=lambda n: d[n]["relative_drop"])
        rows.append([key, f"{positive}/13", f"{mean_abs:.4f}",
                     f"{d[max_node]['relative_drop']:+.3f} ({max_node})"])
    table_page(pdf, "Causal ablation summary per site",
               ["Site", "Nodes w/ + drop", "Mean |rel drop|", "Max rel drop"], rows)

    # Detailed L7 resid table
    d = ablation["L7_resid"]
    rows = []
    for n in NODES:
        v = d[n]
        rows.append([n, str(v["feature"]), f"{v['mean_baseline_prob']:.6f}",
                     f"{v['mean_ablated_prob']:.6f}", f"{v['prob_drop']:+.6f}",
                     f"{v['relative_drop']:+.3f}"])
    table_page(pdf, "Detailed ablation — L7 residual",
               ["Node", "Feature", "Baseline P", "Ablated P", "Prob Drop", "Rel Drop"],
               rows)

    # Heatmap of relative drops (clipped)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(BG_COL)
    for si, site in enumerate(SITES):
        mat = np.zeros((len(LAYERS_4), len(NODES)))
        for li, layer in enumerate(LAYERS_4):
            for ni, node in enumerate(NODES):
                mat[li, ni] = ablation[f"L{layer}_{site}"][node]["relative_drop"]
        ax = axes[si]
        clip = np.clip(mat, -2, 2)
        im = ax.imshow(clip, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(NODES)))
        ax.set_xticklabels(NODES, rotation=55, ha="right", fontsize=8)
        ax.set_yticks(range(len(LAYERS_4)))
        ax.set_yticklabels([f"L{l}" for l in LAYERS_4])
        ax.set_title(f"{site.upper()} — relative drop", fontweight="bold")
        for li in range(len(LAYERS_4)):
            for ni in range(len(NODES)):
                v = mat[li, ni]
                ax.text(ni, li, f"{v:.1f}", ha="center", va="center",
                        fontsize=6, color="white" if abs(clip[li, ni]) > 1 else "black")
    fig.colorbar(im, ax=axes, shrink=0.7, label="Relative drop (clipped ±2)")
    fig.suptitle("Relative probability drop on single-feature ablation",
                 fontweight="bold")
    pdf.savefig(fig); plt.close(fig)

    # Probe-vs-ablation scatter at L7 resid
    probe = load(ADV2 / "linear_probe_L7_resid.json")
    raw_cls = probe["raw_activation_probe"]["per_class_accuracy"]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(BG_COL)
    xs = [raw_cls[n] for n in NODES]
    ys = [d[n]["relative_drop"] for n in NODES]
    ax.scatter(xs, ys, s=80, color=RESID_COL, zorder=3, edgecolor="black")
    for x, y, n in zip(xs, ys, NODES):
        ax.annotate(n, (x, y), fontsize=8, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")
    r = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlabel("Linear probe accuracy (raw, L7 resid)")
    ax.set_ylabel("Relative prob drop on ablation")
    ax.set_title(f"Probe accuracy vs causal effect   (Pearson r = {r:.3f})",
                 fontweight="bold")
    ax.axhline(0, ls="--", color="gray", alpha=0.5)
    ax.grid(color=GRID_COL); ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 5 — Co-activation structure
# =========================================================================
def finding_5_coactivation(pdf):
    finding_title(pdf, 5, "Co-activation Structure",
                  "Do selective features fire in clusters or independently?")

    explanation_page(pdf, "Finding 5 — Explanation", [
        "For each winning feature we record its self-activation rate (how often it fires on its "
        "target node) and its top co-firing partner (which other feature fires most often at the "
        "same time). A high mean top-cofire rate means features come in correlated bundles; a "
        "low one means they fire independently.",

        "Across all 4 late layers × 2 sites, mean top-cofire rates are small (typically well "
        "under 0.3). The SAE has learned features that are mostly independent, not redundant "
        "copies. This is reassuring — it means the ablation results in Finding 4 aren't being "
        "masked by a silent backup feature firing every time the main one fires.",

        "Self-activation rates, in contrast, are high (often >0.6), which confirms that the "
        "features identified as 'selective for node X' really do light up on X."
    ])

    summary_rows = []
    for layer, site in SITES_4:
        key = f"{layer}_{site}"
        coact = try_load(ADV / f"coact_{key}.json")
        if coact is None:
            continue
        selfs, maxcs = [], []
        for n in NODES:
            if n not in coact: continue
            selfs.append(coact[n].get("self_activation_rate", 0))
            top = coact[n]["top_cofiring"]
            maxcs.append(top[0]["coact_rate"] if top else 0)
        regime = ("independent" if np.mean(maxcs) < 0.1
                  else "moderate" if np.mean(maxcs) < 0.3
                  else "clustered")
        summary_rows.append([key, f"{np.mean(selfs):.3f}", f"{np.mean(maxcs):.3f}", regime])

    table_page(pdf, "Co-activation summary",
               ["Site", "Mean self-fire", "Mean max co-fire", "Regime"], summary_rows)

    # Detailed L7 resid table
    coact = load(ADV / "coact_L7_resid.json")
    rows = []
    for n in NODES:
        if n not in coact: continue
        v = coact[n]
        top = v["top_cofiring"][0] if v["top_cofiring"] else {"feature": "—", "coact_rate": 0}
        rows.append([n, str(v["feature"]), f"{v.get('self_activation_rate', 0):.3f}",
                     str(top["feature"]), f"{top['coact_rate']:.3f}"])
    table_page(pdf, "L7 residual — top co-firing partner per node",
               ["Node", "Feature", "Self-fire", "Top partner", "Co-fire rate"], rows)

    # Plot: self vs top-cofire, 8 sites overlaid
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_COL)
    for layer, site in SITES_4:
        coact = try_load(ADV / f"coact_L{layer}_{site}.json")
        if coact is None: continue
        xs, ys = [], []
        for n in NODES:
            if n not in coact: continue
            xs.append(coact[n].get("self_activation_rate", 0))
            top = coact[n]["top_cofiring"]
            ys.append(top[0]["coact_rate"] if top else 0)
        c = MLP_COL if site == "mlp" else RESID_COL
        m = "o" if site == "mlp" else "s"
        ax.scatter(xs, ys, color=c, marker=m, alpha=0.55, s=55,
                   label=f"L{layer} {site}")
    ax.set_xlabel("Self-activation rate")
    ax.set_ylabel("Top co-firing rate")
    ax.set_title("Features fire often but rarely in lockstep", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(color=GRID_COL); ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 6 — Cross-layer similarity
# =========================================================================
def finding_6_cross_layer(pdf):
    finding_title(pdf, 6, "Cross-layer Feature Similarity",
                  "Does a 'For' feature at L5 look like a 'For' feature at L6?")

    explanation_page(pdf, "Finding 6 — Explanation", [
        "For each AST node we line up that node's winning feature at every layer and compute "
        "cosine similarity of their activation patterns on a shared prompt set. If features were "
        "the same concept carried through layers, we'd see high off-diagonal similarity. "
        "Instead we see very low.",

        "On both streams the mean off-diagonal cosine is small — features at different layers "
        "are mostly orthogonal. Adjacent layers show slightly more correlation than distant "
        "ones, but even adjacent-layer similarities rarely exceed 0.3.",

        "The takeaway: each layer develops its own feature vocabulary. There isn't a single "
        "'For' direction that gets copied forward; the model rebuilds its syntactic "
        "representation at each level. That's either because the representation genuinely "
        "changes form (plausible — early layers might mix syntax with tokens, later ones with "
        "semantic context), or because the SAE learns a different basis at each layer and the "
        "same underlying direction ends up split across different latents."
    ])

    for site_type in ["mlp", "resid"]:
        cl = try_load(ADV / f"cross_layer_{site_type}.json")
        if cl is None: continue

        all_mats = []
        node_off = []
        layers = None
        for n in NODES:
            if n not in cl: continue
            mat = np.array(cl[n]["sim_matrix"])
            all_mats.append(mat)
            layers = cl[n]["layers"]
            mask = ~np.eye(mat.shape[0], dtype=bool)
            node_off.append((n, mat[mask].mean()))
        if not all_mats: continue

        avg = np.mean(all_mats, axis=0)
        mask = ~np.eye(avg.shape[0], dtype=bool)

        # Heatmap
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor(BG_COL)
        im = ax.imshow(avg, cmap="RdBu_r", vmin=-0.3, vmax=1.0)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers])
        for i in range(len(layers)):
            for j in range(len(layers)):
                ax.text(j, i, f"{avg[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(avg[i,j]) > 0.5 else "black")
        ax.set_title(f"Mean feature-activation similarity — {site_type.upper()}\n"
                     f"(off-diagonal mean = {avg[mask].mean():.3f})",
                     fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
        save_fig(pdf, fig)

        # Per-node sortedness table
        node_off.sort(key=lambda x: -x[1])
        rows = [[n, f"{s:.4f}"] for n, s in node_off]
        table_page(pdf,
                   f"Cross-layer stability per node — {site_type.upper()}",
                   ["Node", "Mean off-diag cosine"], rows)


# =========================================================================
#  FINDING 7 — Cross-site alignment
# =========================================================================
def finding_7_cross_site(pdf):
    finding_title(pdf, 7, "Cross-site Alignment (MLP ↔ Residual)",
                  "Do MLP-output and residual-stream SAEs recover the same features "
                  "or different ones?")

    explanation_page(pdf, "Finding 7 — Explanation", [
        "For each AST node we take the winning MLP feature and the winning residual feature at "
        "the same layer, and compute their cosine similarity — both the 'assigned' cosine (the "
        "two winners we pre-chose per stream) and the 'best-match' cosine (the closest residual "
        "feature to the MLP winner, over all features at that site).",

        "Assigned cosines hover around 0; best-match cosines are also small. MLP and residual "
        "features encoding the same AST concept are essentially orthogonal — the two sites "
        "encode syntax through completely different directions, not shared representations "
        "copied across locations.",

        "This is somewhat surprising. One might have expected the residual stream to be a sum "
        "that includes the MLP output, so MLP features should reappear in residual space. They "
        "don't — which suggests that either (a) the SAE basis is very sensitive to which "
        "activation family it's trained on, or (b) the MLP writes into directions that are "
        "immediately recombined with attention output in residual space, producing a different "
        "sparse decomposition."
    ])

    for layer in ["L4", "L5", "L6", "L7"]:
        cs = try_load(ADV / f"cross_site_{layer}.json")
        if cs is None: continue
        rows, assigned, bestm = [], [], []
        for n in NODES:
            if n not in cs: continue
            v = cs[n]
            assigned.append(v["assigned_cosine"])
            bestm.append(v["best_resid_cosine"])
            rows.append([n, str(v["mlp_feat"]), str(v["resid_feat"]),
                         f"{v['assigned_cosine']:+.4f}",
                         f"{v['best_resid_cosine']:+.4f}",
                         str(v["best_resid_match"])])
        note = (f"Mean assigned cosine = {np.mean(assigned):+.4f}   |   "
                f"Mean best-match cosine = {np.mean(bestm):+.4f}")
        table_page(pdf, f"Cross-site alignment — {layer}",
                   ["Node", "MLP feat", "Resid feat", "Assigned cos",
                    "Best cos", "Best match"], rows, note=note)

    # Plot — assigned cosine across layers
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_COL)
    layers = ["L4", "L5", "L6", "L7"]
    means_assigned, means_best = [], []
    for layer in layers:
        cs = try_load(ADV / f"cross_site_{layer}.json")
        if cs is None:
            means_assigned.append(np.nan); means_best.append(np.nan); continue
        a = [cs[n]["assigned_cosine"] for n in NODES if n in cs]
        b = [cs[n]["best_resid_cosine"] for n in NODES if n in cs]
        means_assigned.append(np.mean(a))
        means_best.append(np.mean(b))
    x = np.arange(len(layers)); w = 0.35
    ax.bar(x - w/2, means_assigned, w, color="#64748b", label="Assigned (MLP winner ↔ resid winner)")
    ax.bar(x + w/2, means_best, w, color="#f59e0b", label="Best-match (MLP winner ↔ closest resid)")
    ax.set_xticks(x); ax.set_xticklabels(layers)
    ax.set_ylabel("Mean cosine similarity (over 13 AST nodes)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("MLP and residual features are essentially orthogonal",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", color=GRID_COL); ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 8 — Logit lens
# =========================================================================
def finding_8_logit_lens(pdf):
    adv1 = load(RESULTS / "advanced_analysis.json")

    finding_title(pdf, 8, "Logit Lens per Feature",
                  "Do SAE features project onto vocabulary tokens that make "
                  "semantic sense for their AST node?")

    explanation_page(pdf, "Finding 8 — Explanation", [
        "A feature's decoder direction can be pushed through the unembedding to see which "
        "output tokens it promotes and inhibits. If the feature is truly a 'For loop' feature "
        "we would expect tokens like `for`, `range`, `in` to be promoted. This is the logit "
        "lens on individual SAE features.",

        "The result is mixed. Some features show crisp semantic alignment — For features "
        "promote iteration-related tokens, Return features promote 'Return'/'result', Try "
        "features promote 'except'/'error'. Others promote generic programming tokens or "
        "method-call prefixes (`.set`, `.read`, `.strip`) that relate only loosely to the node.",

        "Residual-stream features at L4–L5 give the cleanest matches; later layers become "
        "noisier as the features presumably encode more contextual information. This is the "
        "weakest-evidence finding in the report: it shows a signal but it's a qualitative one "
        "that needs manual judgement to assess."
    ])

    # Top-1 promoted heatmap per site
    for site in SITES:
        fig, ax = plt.subplots(figsize=(13, 5.5))
        fig.patch.set_facecolor(BG_COL)
        mat = np.zeros((len(LAYERS_4), len(NODES)))
        annot = [[""] * len(NODES) for _ in LAYERS_4]
        for li, layer in enumerate(LAYERS_4):
            key = f"logit_lens_L{layer}_{site}"
            if key not in adv1: continue
            for ni, node in enumerate(NODES):
                if node not in adv1[key]: continue
                top = adv1[key][node]["promoted"][0]
                mat[li, ni] = top["logit"]
                annot[li][ni] = top["token"][:7]
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=0.1)
        ax.set_xticks(range(len(NODES)))
        ax.set_xticklabels(NODES, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(LAYERS_4)))
        ax.set_yticklabels([f"L{l}" for l in LAYERS_4])
        ax.set_title(f"Top-1 promoted token per feature — {site.upper()}",
                     fontweight="bold")
        for li in range(len(LAYERS_4)):
            for ni in range(len(NODES)):
                ax.text(ni, li, annot[li][ni], ha="center", va="center",
                        fontsize=6, color="black" if mat[li, ni] < 0.05 else "white")
        fig.colorbar(im, ax=ax, shrink=0.7, label="Top-1 logit")
        save_fig(pdf, fig)

    # Pick notable semantic hits
    kw_map = {
        "for": ["for", "range", "iter", "enum"],
        "while": ["while", "loop"],
        "if": ["if", "else", "elif", "not"],
        "functiondef": ["def", "func", "return"],
        "asyncfunctiondef": ["async", "await", "def"],
        "classdef": ["class", "self", "__init__"],
        "with": ["with", "open", "file"],
        "try": ["try", "except", "error", "raise"],
        "lambda": ["lambda"],
        "return": ["return", "result"],
        "yield": ["yield", "generator"],
        "listcomp": ["list", "comp"],
        "assert": ["assert", "true", "false"],
    }
    rows = []
    for layer in LAYERS_4:
        for site in SITES:
            key = f"logit_lens_L{layer}_{site}"
            if key not in adv1: continue
            for node in NODES:
                if node not in adv1[key]: continue
                info = adv1[key][node]
                for tok_info in info["promoted"][:5]:
                    t = tok_info["token"].lower().strip()
                    if any(kw in t for kw in kw_map.get(node.lower(), [])):
                        rows.append([str(layer), site, node, str(info["feature"]),
                                     tok_info["token"], f"{tok_info['logit']:+.4f}"])
                        break
    rows = rows[:30]
    table_page(pdf, "Semantically aligned promotions (top-5 matches)",
               ["Layer", "Site", "Node", "Feature", "Token", "Logit"], rows)


# =========================================================================
#  FINDING 9 — Steering
# =========================================================================
def finding_9_steering(pdf):
    finding_title(pdf, 9, "Feature Steering",
                  "If we amplify a feature's activation, can we steer the model "
                  "into producing its target keyword?")

    explanation_page(pdf, "Finding 9 — Explanation", [
        "Steering multiplies a single SAE feature's activation by 2×, 5×, or 10× and regenerates "
        "code. The question: does the output contain the AST node's keyword more often?",

        "It mostly doesn't. At all tested multipliers, codegen steering produces the target "
        "keyword at very low rates, and at 10× the generations often collapse into degenerate "
        "text ('!!!!'). This is a negative result.",

        "It doesn't mean the features aren't real — the probe, minimal-pairs and nesting "
        "analyses all say they are. It means these SAE features encode abstract syntactic roles "
        "rather than literal keyword production directions, so amplifying them doesn't push the "
        "model toward emitting a token. Steering at the MLP/residual level is a blunt "
        "intervention; surgical keyword steering may require a much later layer close to the "
        "unembedding."
    ])

    # Summary per site
    rows = []
    for layer, site in SITES_4:
        cg = try_load(ADV / f"steering_codegen_L{layer}_{site}.json")
        if cg is None: continue
        for mult in ["2.0", "5.0", "10.0"]:
            rates = [cg[n][mult].get("keyword_rate", 0)
                     for n in NODES if n in cg and mult in cg[n]]
            rows.append([f"L{layer} {site}", mult,
                         f"{np.mean(rates)*100:.1f}%" if rates else "—"])
    table_page(pdf, "Codegen steering — mean keyword rate",
               ["Site", "Multiplier", "Keyword rate"], rows)

    # Plot — codegen keyword rates grouped
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG_COL)
    mults = ["2.0", "5.0", "10.0"]
    sites = [f"L{l} {s}" for l, s in SITES_4]
    data = {m: [] for m in mults}
    for layer, site in SITES_4:
        cg = try_load(ADV / f"steering_codegen_L{layer}_{site}.json")
        for m in mults:
            if cg is None:
                data[m].append(0); continue
            rates = [cg[n][m].get("keyword_rate", 0)
                     for n in NODES if n in cg and m in cg[n]]
            data[m].append(np.mean(rates) if rates else 0)
    x = np.arange(len(sites)); w = 0.25
    for i, m in enumerate(mults):
        ax.bar(x + (i - 1) * w, data[m], w, label=f"{m}×")
    ax.set_xticks(x); ax.set_xticklabels(sites, rotation=25, ha="right")
    ax.set_ylabel("Mean keyword rate in steered generations")
    ax.set_title("Codegen steering barely produces target keywords",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", color=GRID_COL); ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 10 — Minimal pairs
# =========================================================================
def finding_10_minimal_pairs(pdf):
    finding_title(pdf, 10, "Minimal Pairs",
                  "Does a For-feature fire more on For than on the matched While version "
                  "of the same program?")

    explanation_page(pdf, "Finding 10 — Explanation", [
        "Minimal pairs are tiny code snippets that differ only in the AST node in question — "
        "e.g. a For loop vs. the same loop rewritten as While, FunctionDef vs Lambda, "
        "For vs ListComp. We ask: does the node's winning feature fire more often on the "
        "matching side of the pair than the non-matching side?",

        "Most pairs pass this test across layers — the feature fires preferentially on its "
        "target, not the syntactically-close alternative. This is a stronger form of "
        "selectivity than simply 'fires a lot on For'; it rules out confounds from unrelated "
        "co-occurring tokens because the only difference between pairs is the construct itself.",

        "Together with the nesting result in Finding 11 this is the strongest behavioural "
        "evidence that the features really do encode the AST concept we've labelled them with."
    ])

    for layer, site in SITES_4:
        mp = try_load(ADV2 / f"minimal_pairs_L{layer}_{site}.json")
        if mp is None: continue
        rows = []
        n_sel = 0
        for pair_name, v in mp.items():
            a_node, b_node = v["a_node"], v["b_node"]
            a_on_a = v["a_activations"].get(a_node, {}).get("firing_frac", 0)
            a_on_b = v["b_activations"].get(a_node, {}).get("firing_frac", 0) if v.get("b_activations") else 0
            sel = a_on_a > a_on_b
            n_sel += int(sel)
            rows.append([pair_name, a_node, b_node,
                         f"{a_on_a:.3f}", f"{a_on_b:.3f}",
                         "YES" if sel else "no"])
        table_page(pdf, f"Minimal pairs — L{layer} {site}",
                   ["Pair", "A node", "B node", "Fire on A", "Fire on B", "Selective?"],
                   rows, note=f"{n_sel}/{len(rows)} pairs pass selectivity")


# =========================================================================
#  FINDING 11 — Nesting / compositionality
# =========================================================================
def finding_11_nesting(pdf):
    finding_title(pdf, 11, "Nesting and Compositionality",
                  "When AST constructs nest, do the outer and inner features "
                  "fire together?")

    explanation_page(pdf, "Finding 11 — Explanation", [
        "Python syntax is compositional — for-inside-if, def-inside-class. If our features "
        "really encode node types, nesting one construct inside another should make both "
        "features fire. We test this with hand-crafted nested snippets (for_in_if, "
        "def_in_class, if_in_for, try_in_for, with_in_def, lambda_in_for), measuring whether "
        "both the outer and inner winning features activate above threshold on the nested "
        "program.",

        "Composition holds for the vast majority of configurations at all 4 late layers and "
        "both sites. This is strong evidence for compositional, modular feature semantics: "
        "the SAE is not just memorising surface patterns, it's assembling independent units.",

        "This also validates the rest of the paper's interpretations — if composition had "
        "failed, we'd have to treat 'For feature' and 'If feature' as just convenient labels "
        "for whatever happened to activate on those programs. Because it holds, we can treat "
        "them as genuine representations of the construct."
    ])

    all_rows = []
    for layer, site in SITES_4:
        nest = try_load(ADV2 / f"nesting_L{layer}_{site}.json")
        if nest is None: continue
        n_ok = sum(1 for v in nest.values() if v.get("composition_holds"))
        all_rows.append([f"L{layer} {site}", f"{n_ok}/{len(nest)}"])
    table_page(pdf, "Compositionality summary",
               ["Site", "Configs where both features fire"], all_rows)

    # Detail per-config at L7 resid
    nest = load(ADV2 / "nesting_L7_resid.json")
    rows = []
    for cfg, v in nest.items():
        rows.append([cfg, v["outer_node"], v["inner_node"],
                     f"{v['nested']['outer_feat_max']:.3f}",
                     f"{v['nested']['inner_feat_max']:.3f}",
                     "YES" if v["nested"].get("both_fire") else "no",
                     "YES" if v.get("composition_holds") else "no"])
    table_page(pdf, "Nesting detail — L7 residual",
               ["Config", "Outer", "Inner", "Outer max", "Inner max",
                "Both fire?", "Composes?"], rows)

    # Plot — bar of composition rate per site
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_COL)
    sites, rates, cols = [], [], []
    for layer, site in SITES_4:
        nest = try_load(ADV2 / f"nesting_L{layer}_{site}.json")
        if nest is None: continue
        rate = np.mean([v.get("composition_holds", False) for v in nest.values()])
        sites.append(f"L{layer} {site}")
        rates.append(rate)
        cols.append(MLP_COL if site == "mlp" else RESID_COL)
    ax.bar(range(len(sites)), rates, color=cols)
    ax.set_xticks(range(len(sites)))
    ax.set_xticklabels(sites, rotation=25, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Fraction of nesting configs composing")
    ax.set_title("Compositionality holds across layers and sites",
                 fontweight="bold")
    ax.axhline(1, ls="--", color="gray", alpha=0.5)
    ax.grid(axis="y", color=GRID_COL); ax.set_facecolor(BG_COL)
    ax.legend(handles=[Patch(color=MLP_COL, label="MLP"),
                       Patch(color=RESID_COL, label="Residual")])
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 12 — Positional analysis
# =========================================================================
def finding_12_positional(pdf):
    finding_title(pdf, 12, "Positional Firing",
                  "Do features lock onto a particular sequence position, or fire "
                  "wherever the syntactic role appears?")

    explanation_page(pdf, "Finding 12 — Explanation", [
        "For each feature, we log the relative position (0 = start of sequence, 1 = end) of "
        "every firing event across a large corpus. If features were positional gadgets "
        "(\"fires on token 17\"), the std would be tiny and the mean tightly clustered. They're "
        "not.",

        "Across all late-layer sites, the mean relative position hovers near 0.5 and the "
        "standard deviation is large (typically >0.25). Features fire wherever their target "
        "construct appears, not at fixed positions.",

        "This is exactly the behaviour you want of a genuinely abstract feature. It's also why "
        "the minimal-pairs test is meaningful — if features were positional, the 'For vs While' "
        "pair would separate on position rather than node type."
    ])

    for layer, site in SITES_4:
        pos = try_load(ADV2 / f"positional_L{layer}_{site}.json")
        if pos is None: continue
        rows = []
        for n in NODES:
            if n not in pos: continue
            v = pos[n]
            top = v["top_tokens_at_fire"][0]["token"] if v["top_tokens_at_fire"] else "—"
            rows.append([n, str(v["feature"]), str(v["n_firings"]),
                         f"{v['mean_rel_position']:.3f}",
                         f"{v['std_rel_position']:.3f}",
                         f"{v['median_rel_position']:.3f}",
                         repr(top)[:12]])
        table_page(pdf, f"Positional firing — L{layer} {site}",
                   ["Node", "Feature", "# firings", "Mean pos", "Std pos",
                    "Median", "Top token"], rows, fontsize=7)

    # Plot — mean ± std per site overlaid
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(BG_COL)
    for ii, (layer, site) in enumerate(SITES_4):
        pos = try_load(ADV2 / f"positional_L{layer}_{site}.json")
        if pos is None: continue
        ms = [pos[n]["mean_rel_position"] for n in NODES if n in pos]
        ss = [pos[n]["std_rel_position"] for n in NODES if n in pos]
        x = np.arange(len(ms)) + ii * 0.08
        c = MLP_COL if site == "mlp" else RESID_COL
        ax.errorbar(x, ms, yerr=ss, fmt="o", color=c, alpha=0.6,
                    label=f"L{layer} {site}", ms=5, capsize=2)
    ax.axhline(0.5, ls="--", color="gray", alpha=0.5)
    ax.set_xticks(np.arange(len(NODES)) + 0.15)
    ax.set_xticklabels(NODES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Relative position (mean ± std)")
    ax.set_ylim(-0.1, 1.1)
    ax.set_title("Features are not position-locked", fontweight="bold")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(axis="y", color=GRID_COL); ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 13 — Circuit analysis
# =========================================================================
def finding_13_circuit(pdf):
    master = try_load(REPORTS / "master_circuit_report.json")

    finding_title(pdf, 13, "Circuit Metrics",
                  "Beyond ablation: drop_zero, drop_mean, patch_gain, and the "
                  "upstream attention heads feeding each feature.")

    explanation_page(pdf, "Finding 13 — Explanation", [
        "A circuit report per node gives us three causal metrics and an upstream analysis. "
        "drop_zero and drop_mean are zero- and mean-ablation effects on the matching keyword's "
        "probability; patch_gain is the improvement from patching the feature into an unrelated "
        "prompt. Argmax-changed asks whether the top predicted token actually flips.",

        "Effects are typically small (and often negative — single-feature ablation is not a "
        "reliable way to destroy syntax). Argmax rarely flips. These numbers reinforce the "
        "redundancy picture from Finding 4: syntax is distributed, so deleting one feature "
        "barely moves the top-1.",

        "The upstream analysis is more interesting: across nodes, a small set of attention "
        "heads (typically at earlier layers) show up as dominant upstream inputs to the "
        "circuit-winning features. These heads look like the model's 'syntax routers' that "
        "deliver structural information to the MLP/residual features we analyse."
    ])

    if master is None: return

    # Summary table
    rows = []
    dz, dm, pg = [], [], []
    changed = 0
    nodes_present = [n for n in NODES if n in master]
    for n in nodes_present:
        v = master[n]; m = v["metrics"]
        dz.append(m["drop_zero"]); dm.append(m["drop_mean"]); pg.append(m["patch_gain"])
        if v["argmax"]["changed"]: changed += 1
        rows.append([n, str(v["feature"]), str(len(v.get("all_winners", []))),
                     f"{m['drop_zero']:+.4f}", f"{m['drop_mean']:+.4f}",
                     f"{m['patch_gain']:+.4f}",
                     "YES" if v["argmax"]["changed"] else "no"])
    table_page(pdf, "Master circuit report — per node",
               ["Node", "Feature", "# winners", "drop_zero", "drop_mean",
                "patch_gain", "Argmax flipped?"], rows,
               note=(f"Mean drop_zero={np.mean(dz):+.4f}, "
                     f"drop_mean={np.mean(dm):+.4f}, "
                     f"patch_gain={np.mean(pg):+.4f}, "
                     f"argmax flipped for {changed}/{len(nodes_present)} nodes"))

    # Upstream heads
    head_counts = defaultdict(int)
    head_weights = defaultdict(list)
    for n in nodes_present:
        for up in master[n].get("upstream", []):
            head_counts[up["source"]] += 1
            head_weights[up["source"]].append(up["weight"])
    head_rows = sorted(head_counts.items(), key=lambda x: -x[1])[:12]
    urows = [[h, f"{c}/{len(nodes_present)}", f"{np.mean(head_weights[h]):.4f}"]
             for h, c in head_rows]
    table_page(pdf, "Top upstream attention heads",
               ["Head", "Appears for", "Mean weight"], urows)

    # Plot — circuit metrics per node
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(BG_COL)
    x = np.arange(len(nodes_present)); w = 0.25
    ax.bar(x - w, dz, w, label="drop_zero", color="#e74c3c")
    ax.bar(x,      dm, w, label="drop_mean", color="#3498db")
    ax.bar(x + w, pg, w, label="patch_gain", color="#2ecc71")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(nodes_present, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Metric value")
    ax.set_title("Circuit metrics per AST node", fontweight="bold")
    ax.legend(); ax.grid(axis="y", color=GRID_COL); ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 14 — Parameter sweep robustness
# =========================================================================
def finding_14_sweep(pdf):
    sweep = try_load(SWEEP / "sweep_summary.json")
    finding_title(pdf, 14, "Parameter Sweep Robustness",
                  "Are findings artefacts of particular selectivity thresholds?")

    explanation_page(pdf, "Finding 14 — Explanation", [
        "The pipeline exposes two key knobs for feature discovery: feature_selectivity_threshold "
        "(how selective a feature must be to count) and specificity_ratio (how much more it "
        "must fire on its target than on everything else). We swept 5×3 combinations and "
        "re-ran the discovery plus circuit metrics on each.",

        "Across the sweep the number of 'specific' features varies but the mean drop_zero — the "
        "main circuit effect — is reasonably stable. No single configuration gives a dramatically "
        "different picture. This is the robustness check that the rest of the findings are not "
        "the artefact of a lucky threshold.",

        "The sweep also shows the expected monotone relationships: higher selectivity → fewer "
        "features; higher specificity → fewer features; both → fewer but cleaner."
    ])

    if sweep is None: return
    rows = []
    for r in sweep:
        p = r["params"]
        rows.append([f"{p['sae_params.feature_selectivity_threshold']}",
                     f"{p['sae_params.specificity_ratio']}",
                     str(r["nodes_found"]), str(r["n_specific"]),
                     f"{r['avg_drop_zero']:.4f}"])
    table_page(pdf, "Full sweep table",
               ["Selectivity", "Specificity", "Nodes found", "# specific",
                "Avg drop_zero"], rows)

    # Plot — heatmap of n_specific over sel × spec
    sels = sorted({r["params"]["sae_params.feature_selectivity_threshold"] for r in sweep})
    specs = sorted({r["params"]["sae_params.specificity_ratio"] for r in sweep})
    mat_count = np.zeros((len(sels), len(specs)))
    mat_drop  = np.zeros((len(sels), len(specs)))
    for r in sweep:
        i = sels.index(r["params"]["sae_params.feature_selectivity_threshold"])
        j = specs.index(r["params"]["sae_params.specificity_ratio"])
        mat_count[i, j] = r["n_specific"]
        mat_drop[i, j]  = r["avg_drop_zero"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(BG_COL)
    im1 = ax1.imshow(mat_count, cmap="Greens")
    ax1.set_xticks(range(len(specs))); ax1.set_xticklabels(specs)
    ax1.set_yticks(range(len(sels)));  ax1.set_yticklabels(sels)
    ax1.set_xlabel("specificity_ratio"); ax1.set_ylabel("selectivity_threshold")
    ax1.set_title("# specific features", fontweight="bold")
    for i in range(len(sels)):
        for j in range(len(specs)):
            ax1.text(j, i, f"{int(mat_count[i,j])}", ha="center", va="center", fontsize=9)
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(mat_drop, cmap="coolwarm", vmin=-0.5, vmax=0.5)
    ax2.set_xticks(range(len(specs))); ax2.set_xticklabels(specs)
    ax2.set_yticks(range(len(sels)));  ax2.set_yticklabels(sels)
    ax2.set_xlabel("specificity_ratio"); ax2.set_ylabel("selectivity_threshold")
    ax2.set_title("Avg drop_zero", fontweight="bold")
    for i in range(len(sels)):
        for j in range(len(specs)):
            ax2.text(j, i, f"{mat_drop[i,j]:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im2, ax=ax2, shrink=0.8)
    fig.suptitle("Parameter sweep robustness", fontweight="bold")
    save_fig(pdf, fig)


# =========================================================================
#  FINDING 15 — Multi-site discovery
# =========================================================================
def finding_15_discovery(pdf):
    disco = try_load(RESULTS / "multilayer_multisite_discovery.json")
    finding_title(pdf, 15, "Multi-site Discovery",
                  "Across all layers and sites, which produce the richest pool "
                  "of selective features?")

    explanation_page(pdf, "Finding 15 — Explanation", [
        "The top-level discovery pass sweeps every (layer, site) combination and counts how "
        "many features it finds that are selective for any of the 13 AST nodes at the "
        "configured thresholds. This gives us a global map of 'where does syntactic structure "
        "live?'",

        "Late layers dominate (consistent with Finding 2's phase transition). Within late "
        "layers, MLP sites find more selective features than residual sites at the same depth "
        "— which is a bit misleading, because Finding 3 showed these features are less "
        "faithful. The MLP SAE casts a wider net but the catch is less reliable.",

        "This finding is presented last because it's the map the other findings were drawn "
        "against. Everything downstream — probes, ablations, nesting, minimal pairs — uses the "
        "features discovered here as its starting point."
    ])

    if disco is None: return

    # Summary table
    rows = []
    tuples = []
    for site_key in sorted(disco.keys()):
        total = disco[site_key]["total_selective"]
        tuples.append((site_key, total))
        rows.append([site_key, str(total), f"{total/len(NODES):.2f}"])
    table_page(pdf, "Selective-feature count per site",
               ["Site", "Total selective", "Mean per AST node"], rows)

    # Plot — bar across all 16 sites
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor(BG_COL)
    keys = [t[0] for t in tuples]
    totals = [t[1] for t in tuples]
    colors = [MLP_COL if "mlp" in k else RESID_COL for k in keys]
    ax.bar(range(len(keys)), totals, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Total selective features")
    ax.set_title("Feature discovery across all sites", fontweight="bold")
    ax.legend(handles=[Patch(color=MLP_COL, label="MLP"),
                       Patch(color=RESID_COL, label="Residual")])
    ax.grid(axis="y", color=GRID_COL); ax.set_facecolor(BG_COL)
    save_fig(pdf, fig)

    # Selectivity distribution for best site
    tuples.sort(key=lambda x: -x[1])
    best_key = tuples[0][0]
    rows = []
    for n in NODES:
        node_data = disco[best_key]["nodes"].get(n)
        if not node_data:
            rows.append([n, "—", "—", "—"]); continue
        sels = [f["sel"] for f in node_data.get("top_by_selectivity", [])]
        rows.append([n, str(node_data["n_selective"]),
                     f"{max(sels):.3f}" if sels else "—",
                     f"{np.mean(sels):.3f}" if sels else "—"])
    table_page(pdf, f"Best site ({best_key}) — selectivity per node",
               ["Node", "# selective", "Top sel", "Mean sel"], rows)


# =========================================================================
#  SYNTHESIS PAGE
# =========================================================================
def synthesis(pdf):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(BG_COL)
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.97, "Synthesis — The Story in One Page", ha="center", va="top",
            fontsize=16, fontweight="bold", color=DARK, transform=ax.transAxes)

    text = (
        "TIER 1 — headline findings\n"
        "──────────────────────────\n"
        "  • SAE faithfulness is site-dependent (Finding 3): residual SAEs lose ~1pp\n"
        "    linear-probe accuracy, MLP SAEs lose ~6pp — despite MLP having higher R².\n"
        "    High reconstruction quality is not enough.\n\n"
        "  • Phase transition in feature count (Finding 2): layers 0–3 host almost no\n"
        "    selective features; layers 4–7 host an order of magnitude more. Syntactic\n"
        "    representations emerge abruptly in the second half of the network.\n\n"
        "  • Features compose under nesting (Finding 11): for-in-if programs fire both\n"
        "    the For and If winning features. Together with minimal-pairs, this is\n"
        "    the strongest behavioural evidence that SAE features encode genuine\n"
        "    compositional syntactic concepts.\n\n"
        "TIER 2 — supporting structure\n"
        "──────────────────────────\n"
        "  • MLP and residual features are orthogonal (Finding 7): the two sites\n"
        "    recover different feature directions, not shared ones.\n"
        "  • Three tiers of AST separability (Finding 3): easy / moderate / hard\n"
        "    classes mirror structural co-occurrence patterns.\n"
        "  • Features fire uniformly across positions (Finding 12): they respond to\n"
        "    syntactic role, not sequence position.\n\n"
        "TIER 3 — nuance and caveats\n"
        "──────────────────────────\n"
        "  • Causal single-feature ablation effects are small (Finding 4, 13):\n"
        "    syntax is distributed, not concentrated. Ablating one feature rarely\n"
        "    flips the argmax.\n"
        "  • Steering largely fails (Finding 9): features encode abstract roles, not\n"
        "    keyword-production directions.\n"
        "  • Cross-layer similarity is low (Finding 6): each layer learns its own\n"
        "    feature vocabulary rather than carrying features forward.\n"
        "  • Findings are robust to hyperparameter sweep (Finding 14).\n"
    )
    ax.text(0.04, 0.90, text, transform=ax.transAxes, fontsize=9.5,
            va="top", fontfamily="monospace", color="#1f2937",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#cbd5e1"))
    pdf.savefig(fig); plt.close(fig)


# =========================================================================
#  MAIN
# =========================================================================
def main():
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(OUT_PDF) as pdf:
        cover_page(pdf)
        finding_1_r2(pdf)
        finding_2_feature_counts(pdf)
        finding_3_linear_probe(pdf)
        finding_4_causal_ablation(pdf)
        finding_5_coactivation(pdf)
        finding_6_cross_layer(pdf)
        finding_7_cross_site(pdf)
        finding_8_logit_lens(pdf)
        finding_9_steering(pdf)
        finding_10_minimal_pairs(pdf)
        finding_11_nesting(pdf)
        finding_12_positional(pdf)
        finding_13_circuit(pdf)
        finding_14_sweep(pdf)
        finding_15_discovery(pdf)
        synthesis(pdf)

    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"Wrote {OUT_PDF}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
