"""
Cross-layer dendrogram corroboration.

For every (layer, site) in the study_v2 per-layer reports, check whether the
four dendrogram clusters (cyan/red/yellow/purple) each get bound by a single
SAE winner feature (>= 2 cluster members sharing the same top-1 feature).

Outputs:
  data/study_v2/reports/dendrogram_all_layers.json
  data/study_v2/reports/dendrogram_all_layers.pdf

PDF pages:
  1. Title + method.
  2. Per-site heatmap: rows = dendrogram clusters, cols = layers, cell = max
     within-cluster bundle size (how many cluster members share one feature).
  3. Per-site table: for each (layer, cluster), list the feature(s) and members.
  4. Tight-pair emergence table across layers (which layer first binds each
     dendrogram tight pair).
"""
import os, sys, json
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import resolve_path, data_path

REPORTS = data_path("study_v2/reports")
PER = os.path.join(REPORTS, "per_layer")
TAG = "old_saes"
N_LAYERS = 8
SITES = ["mlp", "resid"]

CLUSTERS = {
    "Cyan (control flow)": ["While", "Break", "Continue", "Try", "If", "For", "Import"],
    "Red (predicates)":    ["Compare", "IfExp", "Assert", "BoolOp"],
    "Yellow (definitions)": ["FunctionDef", "AsyncFunctionDef", "AnnAssign", "ClassDef", "Return"],
    "Purple (mutations/comp/yields)": [
        "Delete", "AugAssign", "Global", "With", "Raise", "ListComp",
        "GeneratorExp", "SetComp", "Subscript", "Lambda",
        "DictComp", "YieldFrom", "Yield",
    ],
}
CLUSTER_COLOR = {
    "Cyan (control flow)": "#4FC3F7",
    "Red (predicates)": "#E57373",
    "Yellow (definitions)": "#FFD54F",
    "Purple (mutations/comp/yields)": "#BA68C8",
}
TIGHT_PAIRS = [
    ("FunctionDef", "AsyncFunctionDef", 0.04),
    ("ListComp",    "GeneratorExp",     0.07),
    ("AnnAssign",   "ClassDef",         0.14),
    ("While",       "Break",            0.20),
    ("Compare",     "IfExp",            0.36),
    ("If",          "For",              0.48),
]


def load_top1(layer, site):
    """Return {node: top-1 feature_id} for one (layer, site) report."""
    path = os.path.join(PER, f"L{layer}_{site}_{TAG}.json")
    rep = json.load(open(path))
    return {node: nd["candidates"][0]["feature"] for node, nd in rep.items()}


def cluster_bindings(top1):
    """For each cluster, group nodes by feature; return
    {cluster: [{feature, members (within-cluster), cross_cluster_members}]}."""
    # invert: feature -> nodes
    feat2nodes = defaultdict(list)
    for node, feat in top1.items():
        feat2nodes[feat].append(node)

    out = {}
    for cname, members in CLUSTERS.items():
        rows = []
        for feat, nodes in feat2nodes.items():
            inside = [n for n in nodes if n in members]
            if len(inside) >= 2:
                outside = [n for n in nodes if n not in members]
                rows.append({"feature": feat, "within": inside, "cross": outside})
        # sort by bundle size desc
        rows.sort(key=lambda r: -len(r["within"]))
        out[cname] = rows
    return out


def tight_pair_bindings(top1):
    """For each tight pair, record whether both nodes share the same top-1 feat."""
    rows = []
    for a, b, d in TIGHT_PAIRS:
        fa, fb = top1.get(a), top1.get(b)
        rows.append({"pair": [a, b], "cosine_dist": d,
                     "feat_a": fa, "feat_b": fb,
                     "bound": fa is not None and fa == fb})
    return rows


def main():
    all_data = {"layers": {}, "tight_pairs_by_site": {s: [] for s in SITES}}
    for site in SITES:
        for layer in range(N_LAYERS):
            top1 = load_top1(layer, site)
            key = f"L{layer}_{site}"
            all_data["layers"][key] = {
                "cluster_bindings": cluster_bindings(top1),
                "tight_pairs": tight_pair_bindings(top1),
                "top1": top1,
            }
            all_data["tight_pairs_by_site"][site].append({
                "layer": layer,
                "pairs": all_data["layers"][key]["tight_pairs"],
            })

    out_json = os.path.join(REPORTS, "dendrogram_all_layers.json")
    with open(out_json, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Wrote {out_json}")

    # ----------------- PDF -----------------
    pdf_path = os.path.join(REPORTS, "dendrogram_all_layers.pdf")
    cluster_names = list(CLUSTERS.keys())

    with PdfPages(pdf_path) as pdf:
        # Page 1: title + method
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Dendrogram ↔ SAE corroboration across ALL layers",
                     fontsize=16, weight="bold", pad=20)
        lines = [
            "For every transformer layer (0–7) and SAE site (MLP, residual),",
            "we take each AST construct's top-1 'winner' SAE feature and ask:",
            "",
            "  • Bundle size per cluster = largest number of dendrogram-cluster",
            "    members whose top-1 winner is the SAME SAE feature.",
            "    ≥2 means that cluster is 'bound' by a polysemantic feature at that",
            "    (layer, site).",
            "",
            "  • Tight-pair bound = the two specific nodes in a dendrogram",
            "    tight-pair share a top-1 winner (at that layer, site).",
            "",
            "This turns the single-layer corroboration (L7) into a depth profile,",
            "showing whether structure in the dendrogram is encoded consistently",
            "across the stack or only emerges at specific layers.",
        ]
        for i, line in enumerate(lines):
            ax.text(0.05, 0.90 - i * 0.035, line, fontsize=11)
        pdf.savefig(fig); plt.close(fig)

        # Page 2+3: per-site heatmap
        for site in SITES:
            fig, ax = plt.subplots(figsize=(11, 6))
            M = np.zeros((len(cluster_names), N_LAYERS), dtype=int)
            for li in range(N_LAYERS):
                cb = all_data["layers"][f"L{li}_{site}"]["cluster_bindings"]
                for ci, cname in enumerate(cluster_names):
                    rows = cb[cname]
                    M[ci, li] = len(rows[0]["within"]) if rows else 0

            im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=max(M.max(), 3),
                            aspect="auto")
            ax.set_xticks(range(N_LAYERS))
            ax.set_xticklabels([f"L{l}" for l in range(N_LAYERS)])
            ax.set_yticks(range(len(cluster_names)))
            ax.set_yticklabels(cluster_names, fontsize=10)
            ax.set_title(f"{site.upper()} SAE — max within-cluster bundle size per layer",
                         fontsize=13, weight="bold")
            for ci in range(len(cluster_names)):
                for li in range(N_LAYERS):
                    txt = str(M[ci, li]) if M[ci, li] > 0 else ""
                    color = "white" if M[ci, li] >= M.max() * 0.6 else "black"
                    ax.text(li, ci, txt, ha="center", va="center",
                            color=color, fontsize=11, weight="bold")
            plt.colorbar(im, ax=ax, label="cluster members sharing one feature")
            plt.tight_layout()
            pdf.savefig(fig); plt.close(fig)

        # Page 4+5: per-site binding details
        for site in SITES:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.axis("off")
            ax.set_title(f"{site.upper()} SAE — polysemantic bundles per cluster, per layer",
                         fontsize=13, weight="bold", pad=10)
            y = 0.95
            for li in range(N_LAYERS):
                ax.text(0.02, y, f"Layer {li}", fontsize=11, weight="bold",
                        color="#37474F")
                y -= 0.025
                cb = all_data["layers"][f"L{li}_{site}"]["cluster_bindings"]
                any_found = False
                for cname in cluster_names:
                    rows = cb[cname]
                    if not rows:
                        continue
                    for r in rows[:2]:  # top 2 bundles per cluster
                        any_found = True
                        cross = f"  [+cross: {','.join(r['cross'])}]" if r["cross"] else ""
                        txt = f"   {cname:34s}  f{r['feature']}  ← {', '.join(r['within'])}{cross}"
                        ax.text(0.02, y, txt, fontsize=8.5, family="monospace",
                                color=CLUSTER_COLOR[cname])
                        y -= 0.022
                if not any_found:
                    ax.text(0.04, y, "   (no within-cluster bundles)",
                            fontsize=8.5, style="italic", color="#888")
                    y -= 0.022
                y -= 0.008
                if y < 0.05:
                    pdf.savefig(fig); plt.close(fig)
                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.axis("off"); y = 0.95
            pdf.savefig(fig); plt.close(fig)

        # Page: tight-pair emergence table
        for site in SITES:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.axis("off")
            ax.set_title(f"{site.upper()} — tight dendrogram pairs: layers where the pair "
                         "shares a top-1 SAE feature", fontsize=12, weight="bold", pad=10)
            headers = ["Pair", "Cos.dist."] + [f"L{l}" for l in range(N_LAYERS)]
            rows = []
            for (a, b, d) in TIGHT_PAIRS:
                row = [f"{a} ↔ {b}", f"{d:.2f}"]
                for li in range(N_LAYERS):
                    tp = all_data["layers"][f"L{li}_{site}"]["tight_pairs"]
                    hit = next((t for t in tp if t["pair"] == [a, b]), None)
                    row.append("✓" if hit and hit["bound"] else "")
                rows.append(row)
            tab = ax.table(cellText=[headers] + rows, loc="center",
                            cellLoc="center", bbox=[0.02, 0.2, 0.96, 0.6])
            tab.auto_set_font_size(False); tab.set_fontsize(10)
            for i in range(len(headers)):
                tab[(0, i)].set_facecolor("#37474F")
                tab[(0, i)].set_text_props(color="white", weight="bold")
            for ri, row in enumerate(rows, start=1):
                for ci, val in enumerate(row):
                    if ci >= 2 and val == "✓":
                        tab[(ri, ci)].set_facecolor("#C8E6C9")
                    elif ci >= 2:
                        tab[(ri, ci)].set_facecolor("#FFCDD2")
            pdf.savefig(fig); plt.close(fig)

    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
