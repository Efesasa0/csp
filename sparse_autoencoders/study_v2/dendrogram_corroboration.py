"""
Generate a PDF corroborating dendrogram clusters against MLP and resid SAE findings.

Produces:
  data/study_v2/reports/dendrogram_corroboration.pdf
  data/study_v2/reports/dendrogram_corroboration.json
"""
import os, sys, json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import resolve_path, data_path

OUT_DIR = data_path("study_v2/reports")
os.makedirs(OUT_DIR, exist_ok=True)
PDF_PATH = os.path.join(OUT_DIR, "dendrogram_corroboration.pdf")
JSON_PATH = os.path.join(OUT_DIR, "dendrogram_corroboration.json")

# -------- dendrogram clusters from the user-supplied figure --------
CLUSTERS = {
    "Cyan (control flow)":   ["While", "Break", "Continue", "Try", "If", "For", "Import"],
    "Red (predicates)":       ["Compare", "IfExp", "Assert", "BoolOp"],
    "Yellow (definitions)":   ["FunctionDef", "AsyncFunctionDef", "AnnAssign", "ClassDef", "Return"],
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

def cluster_of(node):
    for name, members in CLUSTERS.items():
        if node in members:
            return name
    return "Unclustered"


def shared(rep):
    """Return {feature_id: [nodes]} groups of size>=2."""
    nodes = [n for n in rep if not n.startswith("_")]
    d = defaultdict(list)
    for n in nodes:
        d[rep[n]["feature"]].append(n)
    return {f: ns for f, ns in d.items() if len(ns) > 1}


def main():
    mlp = json.load(open(data_path("reports/circuit_report_L7_mlp.json")))
    res = json.load(open(data_path("reports/circuit_report_L7_resid.json")))
    fac = json.load(open(data_path("reports/factorial_specificity.json")))

    mlp_share = shared(mlp)
    res_share = shared(res)

    # --------- tight-pair corroboration ---------
    tight_report = []
    for a, b, dist in TIGHT_PAIRS:
        mlp_hit = next(((f, ns) for f, ns in mlp_share.items()
                         if a in ns and b in ns), None)
        res_hit = next(((f, ns) for f, ns in res_share.items()
                         if a in ns and b in ns), None)
        tight_report.append({
            "pair": [a, b], "cosine_distance": dist,
            "mlp_feature":  mlp_hit[0] if mlp_hit else None,
            "mlp_group":    mlp_hit[1] if mlp_hit else None,
            "resid_feature": res_hit[0] if res_hit else None,
            "resid_group":   res_hit[1] if res_hit else None,
        })

    # --------- cluster-level corroboration ---------
    def cluster_summary(share):
        rows = []
        for cname, members in CLUSTERS.items():
            features_in = {}
            for f, ns in share.items():
                within = [n for n in ns if n in members]
                if len(within) >= 2:
                    features_in[f] = {
                        "within_cluster": within,
                        "cross_cluster": [n for n in ns if n not in members],
                    }
            rows.append({"cluster": cname, "shared_features": features_in})
        return rows

    mlp_clusters = cluster_summary(mlp_share)
    res_clusters = cluster_summary(res_share)

    # --------- factorial validation per node ---------
    def per_node(site_rep, site_fac):
        out = []
        for node, entry in site_rep.items():
            if node.startswith("_"):
                continue
            fac_e = site_fac.get(node, {})
            out.append({
                "node": node,
                "feature": entry["feature"],
                "classification": fac_e.get("classification", {}).get("primary", "?"),
                "surface":        fac_e.get("classification", {}).get("surface", "?"),
                "cluster": cluster_of(node),
            })
        return out

    per_mlp = per_node(mlp, fac.get("mlp", {}))
    per_res = per_node(res, fac.get("resid", {}))

    report = {
        "dendrogram_clusters": CLUSTERS,
        "tight_pairs": tight_report,
        "mlp_cluster_corroboration":   mlp_clusters,
        "resid_cluster_corroboration": res_clusters,
        "mlp_per_node":   per_mlp,
        "resid_per_node": per_res,
    }
    with open(JSON_PATH, "w") as f:
        json.dump(report, f, indent=2)

    # --------- PDF ---------
    with PdfPages(PDF_PATH) as pdf:
        # Page 1: title + tight-pair table
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Dendrogram ↔ SAE feature-sharing corroboration (L7)",
                     fontsize=16, weight="bold", pad=20)
        lines = [
            "Two independent views of AST-construct structure are compared:",
            "  • Hierarchical clustering of mean residual activations (cosine distance).",
            "  • SAE winner features per construct (MLP and residual SAEs at layer 7).",
            "",
            "A dendrogram tight pair is 'corroborated' when the two nodes share the same",
            "SAE winner feature (a polysemantic bundle). A cluster is corroborated when its",
            "members share features more than they share with out-of-cluster nodes.",
        ]
        for i, line in enumerate(lines):
            ax.text(0.03, 0.93 - i * 0.035, line, fontsize=11)

        # Tight pair table
        headers = ["Pair", "Cos.dist.", "MLP feature", "Resid feature"]
        rows = [[f"{a} ↔ {b}", f"{d:.2f}",
                 f"f{e['mlp_feature']}" if e["mlp_feature"] is not None else "—",
                 f"f{e['resid_feature']}" if e["resid_feature"] is not None else "—"]
                for (a, b, d), e in zip(TIGHT_PAIRS, tight_report)]
        tab = ax.table(cellText=[headers] + rows,
                        loc="center", cellLoc="center",
                        bbox=[0.05, 0.25, 0.9, 0.35])
        tab.auto_set_font_size(False); tab.set_fontsize(11)
        for i in range(len(headers)):
            tab[(0, i)].set_facecolor("#37474F")
            tab[(0, i)].set_text_props(color="white", weight="bold")
        for r_i, row in enumerate(rows, start=1):
            ok = row[2] != "—" or row[3] != "—"
            color = "#C8E6C9" if ok else "#FFCDD2"
            for ci in range(4):
                tab[(r_i, ci)].set_facecolor(color)

        ax.text(0.05, 0.18, "Green = recovered by at least one SAE site. "
                              "Red = no shared SAE feature binds the pair.",
                 fontsize=10, style="italic")
        pdf.savefig(fig); plt.close(fig)

        # Page 2: MLP feature-sharing, colored by cluster
        def sharing_page(title, share, per_node_list):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.set_title(title, fontsize=15, weight="bold")
            ax.set_xlim(0, 10); ax.set_ylim(0, len(share) + 2); ax.axis("off")
            groups = list(share.items())
            groups.sort(key=lambda kv: -len(kv[1]))
            y = len(groups) + 1
            ax.text(0.1, y + 0.3, "Feature", fontsize=11, weight="bold")
            ax.text(1.3, y + 0.3, "Nodes in bundle (colored by dendrogram cluster)",
                     fontsize=11, weight="bold")
            for f, ns in groups:
                ax.text(0.1, y, f"f{f}", fontsize=11, family="monospace")
                x = 1.3
                for n in ns:
                    c = CLUSTER_COLOR.get(cluster_of(n), "#B0BEC5")
                    ax.add_patch(plt.Rectangle((x, y - 0.25), 1.5, 0.5,
                                                facecolor=c, edgecolor="black"))
                    ax.text(x + 0.75, y, n, fontsize=9, ha="center", va="center")
                    x += 1.6
                y -= 1
            # Legend
            handles = [Patch(color=col, label=name) for name, col in CLUSTER_COLOR.items()]
            ax.legend(handles=handles, loc="lower center",
                       ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))
            pdf.savefig(fig); plt.close(fig)

        sharing_page("MLP SAE feature-sharing groups (L7)", mlp_share, per_mlp)
        sharing_page("Residual SAE feature-sharing groups (L7)", res_share, per_res)

        # Page 4: per-node classification tables
        def node_page(title, per_list):
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.set_title(title, fontsize=14, weight="bold")
            ax.axis("off")
            headers = ["Node", "Cluster", "Winner", "Classification", "Surface"]
            rows = []
            for e in sorted(per_list, key=lambda x: (x["cluster"], x["node"])):
                rows.append([e["node"], e["cluster"].split(" ")[0],
                             f"f{e['feature']}",
                             e["classification"],
                             e["surface"] or "—"])
            tab = ax.table(cellText=[headers] + rows,
                            loc="upper center", cellLoc="center",
                            bbox=[0.02, 0.02, 0.96, 0.92])
            tab.auto_set_font_size(False); tab.set_fontsize(9)
            for i in range(len(headers)):
                tab[(0, i)].set_facecolor("#37474F")
                tab[(0, i)].set_text_props(color="white", weight="bold")
            for ri, row in enumerate(rows, start=1):
                cname = next((k for k in CLUSTER_COLOR if k.startswith(row[1])), None)
                color = CLUSTER_COLOR.get(cname, "#ECEFF1")
                for ci in range(5):
                    tab[(ri, ci)].set_facecolor(color)
                # Classification coloring
                cls = row[3]
                if cls == "AST-specific":
                    tab[(ri, 3)].set_facecolor("#A5D6A7")
                elif cls == "Builtin-specific":
                    tab[(ri, 3)].set_facecolor("#EF9A9A")
                elif cls in ("Ambiguous", "Both-respond"):
                    tab[(ri, 3)].set_facecolor("#FFE082")
                if row[4] == "Surface-token":
                    tab[(ri, 4)].set_facecolor("#EF9A9A")
                elif row[4] == "Concept":
                    tab[(ri, 4)].set_facecolor("#A5D6A7")
            pdf.savefig(fig); plt.close(fig)

        node_page("MLP — per-node winner features, clusters, and factorial labels", per_mlp)
        node_page("Residual — per-node winner features, clusters, and factorial labels", per_res)

        # Page 6: corroboration summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.set_title("Corroboration summary: dendrogram ↔ SAE", fontsize=14, weight="bold")

        lines = ["", "Tight pairs (dendrogram distance ≤ 0.5):", ""]
        for e in tight_report:
            a, b = e["pair"]
            mlp_yes = "✓" if e["mlp_feature"] is not None else "✗"
            res_yes = "✓" if e["resid_feature"] is not None else "✗"
            lines.append(f"  {a} ↔ {b}  (d={e['cosine_distance']:.2f})   "
                          f"MLP {mlp_yes}     Resid {res_yes}")
        lines += ["", "Cluster-level bundles (≥2 within-cluster nodes share a feature):"]
        for site_label, site_rows in [("MLP", mlp_clusters), ("Resid", res_clusters)]:
            lines.append(f"  {site_label}:")
            for r in site_rows:
                if r["shared_features"]:
                    for f, d in r["shared_features"].items():
                        extra = ""
                        if d["cross_cluster"]:
                            extra = f"  [+cross: {', '.join(d['cross_cluster'])}]"
                        lines.append(f"    {r['cluster']}: f{f} binds "
                                      f"{', '.join(d['within_cluster'])}{extra}")
                else:
                    lines.append(f"    {r['cluster']}: none")

        lines += ["", "Disagreements flagged by factorial confound test:"]
        for e in per_mlp + per_res:
            if e["classification"] in ("Builtin-specific", "Ambiguous") or e["surface"] == "Surface-token":
                lines.append(f"  [{'MLP' if e in per_mlp else 'Resid'}] "
                              f"{e['node']:16s} → f{e['feature']}  "
                              f"{e['classification']}"
                              + (f" / {e['surface']}" if e["surface"] else ""))

        for i, line in enumerate(lines):
            ax.text(0.03, 0.95 - i * 0.022, line,
                     fontsize=9, family="monospace")
        pdf.savefig(fig); plt.close(fig)

    print(f"Wrote {PDF_PATH}")
    print(f"Wrote {JSON_PATH}")


if __name__ == "__main__":
    main()
