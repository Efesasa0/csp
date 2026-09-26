#!/usr/bin/env python3
"""Generate a comprehensive PDF report of all SAE mechanistic interpretability findings."""

import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import resolve_path, data_path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


RESULTS_DIR = data_path("results")
OUT_PATH = resolve_path("SAE_Analysis_Report.pdf")


def load_json(name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as f:
        return json.load(f)


def add_text_page(pdf, title, lines, fontsize=10):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.05, 0.95, title, fontsize=16, fontweight="bold", va="top")
    text = "\n".join(lines)
    fig.text(0.05, 0.88, text, fontsize=fontsize, va="top", family="monospace",
             wrap=True, transform=fig.transFigure)
    plt.axis("off")
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def add_image_page(pdf, title, image_paths, caption=""):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.97, title, fontsize=14, fontweight="bold", ha="center", va="top")
    if caption:
        fig.text(0.5, 0.93, caption, fontsize=9, ha="center", va="top",
                 style="italic", wrap=True)

    valid_paths = [p for p in image_paths if os.path.exists(p)]
    if not valid_paths:
        fig.text(0.5, 0.5, "[Images not found]", ha="center", fontsize=12)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)
        return

    if len(valid_paths) == 1:
        ax = fig.add_axes([0.05, 0.02, 0.9, 0.88])
        img = Image.open(valid_paths[0])
        ax.imshow(img)
        ax.axis("off")
    else:
        for i, p in enumerate(valid_paths[:2]):
            ax = fig.add_axes([0.02 + i * 0.49, 0.02, 0.47, 0.88])
            img = Image.open(p)
            ax.imshow(img)
            ax.axis("off")

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def add_quad_image_page(pdf, title, image_paths, subtitles=None, caption=""):
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.97, title, fontsize=14, fontweight="bold", ha="center", va="top")
    if caption:
        fig.text(0.5, 0.93, caption, fontsize=9, ha="center", va="top", style="italic")

    positions = [
        [0.02, 0.47, 0.47, 0.44],
        [0.51, 0.47, 0.47, 0.44],
        [0.02, 0.02, 0.47, 0.44],
        [0.51, 0.02, 0.47, 0.44],
    ]

    for i, p in enumerate(image_paths[:4]):
        if not os.path.exists(p):
            continue
        ax = fig.add_axes(positions[i])
        img = Image.open(p)
        ax.imshow(img)
        ax.axis("off")
        if subtitles and i < len(subtitles):
            ax.set_title(subtitles[i], fontsize=8)

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def img(name):
    return os.path.join(RESULTS_DIR, name)


def main():
    retrain = load_json("retrain_summary.json")
    discovery = load_json("multilayer_multisite_discovery.json")
    adv1 = load_json("advanced_analysis.json")
    adv2 = load_json("advanced_analysis_2.json")

    with PdfPages(OUT_PATH) as pdf:

        # ── TITLE PAGE ──
        add_text_page(pdf, "", [
            "",
            "",
            "         Mechanistic Interpretability of Syntax Processing",
            "           in a Code-Generation Transformer via SAEs",
            "",
            "         Model: openai/circuit-sparsity (419M params, 8 layers)",
            "         Method: TopK Sparse Autoencoders on MLP + Residual Stream",
            "         Target: 13 Python AST node types",
            "",
            "",
            "  Analyses:",
            "    1. SAE Training & Quality (R\u00b2 across 16 SAEs)",
            "    2. Inverted Hourglass: Syntax specialization across layers",
            "    3. MLP vs Residual Stream: Disjoint syntax subspaces",
            "    4. Cross-Layer Tracking: Feature persistence across depth",
            "    5. Co-activation: Semantic feature clusters",
            "    6. Logit Lens: Vocabulary projection of syntax features",
            "    7. Causal Zero-Ablation: Feature necessity for keyword prediction",
            "    8. Compositional Nesting: Feature composition under nesting",
            "    9. Linear Probe: SAE latent vs raw activation classification",
            "   10. Minimal Pairs: Controlled syntax discrimination",
            "   11. Feature Steering: Negative result (features detect, don't generate)",
        ], fontsize=11)

        # ── 1. SAE TRAINING QUALITY ──
        r2_lines = ["SAE Training Results: R\u00b2 across 8 layers \u00d7 2 activation sites", "",
                     "  Layer    MLP R\u00b2     Resid R\u00b2    MLP Expansion   Resid Expansion",
                     "  " + "-" * 65]
        mlp_r2s, resid_r2s = [], []
        for layer in range(8):
            mr2 = retrain.get(f"L{layer}_mlp", {}).get("r2", 0)
            rr2 = retrain.get(f"L{layer}_resid", {}).get("r2", 0)
            mlp_r2s.append(mr2)
            resid_r2s.append(rr2)
            r2_lines.append(f"    {layer}      {mr2:.4f}      {rr2:.4f}        8x (k=32)       16x (k=64)")

        r2_lines.extend(["  " + "-" * 65,
            f"  Mean    {np.mean(mlp_r2s):.4f}      {np.mean(resid_r2s):.4f}",
            "",
            "  MLP SAEs: 8x expansion, k=32, 100 epochs, cosine LR with 10-epoch warmup",
            "  Resid SAEs: 16x expansion, k=64, 150 epochs, cosine LR with 15-epoch warmup",
            "  Training data: 8000 code prompts from contrastive AST dataset",
            "  Best-state checkpointing: saved weights at epoch with highest R\u00b2",
            "",
            "  Key observation: R\u00b2 peaks at L4 for both sites (MLP: 0.959, Resid: 0.922),",
            "  suggesting mid-network activations are most compressible by sparse features."])
        add_text_page(pdf, "1. SAE Training Quality", r2_lines)

        # ── 2. INVERTED HOURGLASS ──
        hg_lines = [
            "Feature selectivity counts per (layer, site) \u2014 features with selectivity > 0.10", "",
            "  Layer    MLP selective    Resid selective",
            "  " + "-" * 45,
        ]
        for layer in range(8):
            ms = discovery.get(f"L{layer}_mlp", {}).get("total_selective", 0)
            rs = discovery.get(f"L{layer}_resid", {}).get("total_selective", 0)
            hg_lines.append(f"    {layer}         {ms:>4d}             {rs:>4d}")

        hg_lines.extend(["", "",
            "  The 'inverted hourglass' pattern: syntax specialization peaks mid-network",
            "  (MLP peaks at L4 with 531 features, Resid peaks at L6 with 581 features),",
            "  NOT at the output layer. Early layers (L0-L1) have few selective features;",
            "  the model builds up syntax representations gradually, peaks in the middle,",
            "  then transitions toward token-prediction representations at L7.",
        ])
        add_text_page(pdf, "2. Inverted Hourglass: Syntax Specialization Across Layers", hg_lines)

        add_image_page(pdf, "Inverted Hourglass Visualization",
                        [img("hourglass_analysis.png")],
                        "Feature selectivity counts across 8 layers for both MLP and residual stream sites.")

        add_image_page(pdf, "Feature Landscape & Sharing Matrix",
                        [img("feature_landscape.png"), img("feature_sharing_matrix.png")],
                        "Left: Feature landscape across nodes. Right: Feature sharing between AST node types.")

        # ── 3. CROSS-SITE ALIGNMENT ──
        cs_lines = [
            "Cosine similarity between MLP and Resid decoder directions for the same AST node.", "",
            "  If MLP and Resid encode syntax the same way, cosine should be high (>0.5).",
            "  If they use independent subspaces, cosine should be near zero.", "",
            "  Node                L4 assigned_cos    L4 best_cos    L7 assigned_cos    L7 best_cos",
            "  " + "-" * 80,
        ]
        cs4 = adv1.get("cross_site_L4", {})
        cs7 = adv1.get("cross_site_L7", {})
        for node in sorted(cs4.keys()):
            i4 = cs4.get(node, {})
            i7 = cs7.get(node, {})
            cs_lines.append(f"  {node:20s}     {i4.get('assigned_cosine',0):+.3f}          "
                            f"{i4.get('best_resid_cosine',0):.3f}          "
                            f"{i7.get('assigned_cosine',0):+.3f}          "
                            f"{i7.get('best_resid_cosine',0):.3f}")

        cs_lines.extend(["", "",
            "  Result: All assigned cosine similarities are near zero (<0.1).",
            "  Even the best-matching resid feature reaches only 0.28 (L7 FunctionDef).",
            "",
            "  Conclusion: MLP and residual stream encode syntax in DISJOINT SUBSPACES.",
            "  They are complementary channels, not redundant representations.",
        ])
        add_text_page(pdf, "3. MLP vs Residual Stream: Disjoint Syntax Subspaces", cs_lines)

        add_image_page(pdf, "Cross-Site Alignment Heatmaps",
                        [img("advanced_cross_site_L4.png"), img("advanced_cross_site_L7.png")],
                        "Cosine similarity between MLP and Resid decoder directions at L4 (left) and L7 (right).")

        # ── 4. CROSS-LAYER TRACKING ──
        cl_lines = [
            "Max cross-layer cosine similarity for each AST node's best feature.", "",
            "  Node                MLP max_cos (pair)         Resid max_cos (pair)",
            "  " + "-" * 70,
        ]
        cl_mlp = adv1.get("cross_layer_mlp", {})
        cl_resid = adv1.get("cross_layer_resid", {})
        def get_max_sim(info):
            layers = info.get("layers", [])
            sim = info.get("sim_matrix", [])
            mx, pair = 0, ""
            for i in range(len(layers)):
                for j in range(i + 1, len(layers)):
                    if abs(sim[i][j]) > abs(mx):
                        mx = sim[i][j]
                        pair = f"L{layers[i]}-L{layers[j]}"
            return mx, pair

        for node in sorted(set(list(cl_mlp.keys()) + list(cl_resid.keys()))):
            mlp_mx, mlp_pair = get_max_sim(cl_mlp.get(node, {}))
            res_mx, res_pair = get_max_sim(cl_resid.get(node, {}))
            cl_lines.append(f"  {node:20s}  {mlp_mx:+.3f} ({mlp_pair:>6s})          {res_mx:+.3f} ({res_pair:>6s})")

        cl_lines.extend(["", "",
            "  MLP features: max cross-layer cosine ~0.09 (near orthogonal).",
            "  Each MLP layer computes a completely independent direction for syntax.",
            "",
            "  Resid features: some persist across adjacent layers:",
            "    ClassDef: 0.621 (L4-L5), Yield: 0.597 (L4-L5), ListComp: 0.486 (L1-L2)",
            "  The residual stream acts as a 'communication bus' \u2014 directions written",
            "  at one layer survive into the next 1-2 layers before being transformed.",
        ])
        add_text_page(pdf, "4. Cross-Layer Tracking: Feature Persistence", cl_lines)

        add_image_page(pdf, "Cross-Layer Similarity Matrices",
                        [img("advanced_cross_layer_mlp.png"), img("advanced_cross_layer_resid.png")],
                        "MLP (left): pure diagonal = layer-independent. Resid (right): off-diagonal warmth = persistence.")

        # ── 5. CO-ACTIVATION ──
        add_image_page(pdf, "5. Co-activation: Semantic Feature Clusters",
                        [img("advanced_coactivation_L4_mlp.png"), img("advanced_coactivation_L7_mlp.png")],
                        "Off-diagonal co-firing rates with node labels. "
                        "L4/mlp (left): expression-level cluster. L7/mlp (right): Lambda+Assert, For+Try.")

        add_text_page(pdf, "Co-activation Interpretation", [
            "Features that fire together reveal semantic groupings beyond individual labels.", "",
            "L4/mlp clusters:",
            "  - Expression-level: Yield (6220) + Assert (482) + Lambda (826) co-fire at 0.25-0.35",
            "  - Block-level: For + While share some co-firing",
            "  - If (7823) is relatively isolated", "",
            "L7/mlp clusters:",
            "  - Lambda (4402) + Assert (4714) co-fire at 0.33",
            "  - For (362) + Try (5785) co-fire at 0.22",
            "  - ListComp and Return are isolated", "",
            "Residual stream features show sparser, more independent co-firing patterns.", "",
            "Key finding: MLP features co-fire in interpretable semantic clusters",
            "(expression-level vs block-level constructs), while residual stream features",
            "are more independent. MLP outputs encode compositional syntax structure.",
        ])

        add_image_page(pdf, "Co-activation: Residual Stream",
                        [img("advanced_coactivation_L4_resid.png"), img("advanced_coactivation_L7_resid.png")],
                        "L4/resid (left): sparse co-firing. L7/resid (right): richer clusters.")

        # ── 6. LOGIT LENS ──
        ll_lines = [
            "Project SAE decoder directions through the unembedding matrix to see which",
            "tokens each syntax feature promotes. 'Hit' = target keyword in top 10.", "",
            "  Node              L4/mlp    L4/resid   L7/mlp    L7/resid",
            "  " + "-" * 60,
        ]
        for node in ["For", "While", "If", "FunctionDef", "ClassDef", "Return", "Lambda", "ListComp", "Try"]:
            parts = []
            for lk in ["logit_lens_L4_mlp", "logit_lens_L4_resid", "logit_lens_L7_mlp", "logit_lens_L7_resid"]:
                info = adv1.get(lk, {}).get(node, {})
                promoted = info.get("promoted", [])
                kw = node.lower().replace("functiondef", "def").replace("classdef", "class").replace("listcomp", "[")
                rank = None
                for i, t in enumerate(promoted[:10]):
                    tok = t["token"].strip().lower()
                    if kw in tok or tok in kw:
                        rank = i + 1
                        break
                parts.append(f"rank {rank}" if rank else "  --  ")
            ll_lines.append(f"  {node:18s}  {'  '.join(f'{p:>8s}' for p in parts)}")

        ll_lines.extend(["", "",
            "  L7/mlp has the strongest vocabulary alignment (For at rank 3, Return rank 4).",
            "  Most features promote the target keyword weakly (rank 3-10), not at rank 1.",
            "  Features capture syntax CONTEXT, not the keyword token itself.",
        ])
        add_text_page(pdf, "6. Logit Lens: Vocabulary Projection", ll_lines)

        add_quad_image_page(pdf, "Logit Lens: Top Promoted Tokens per Feature",
                             [img("advanced_logit_lens_L4_mlp.png"),
                              img("advanced_logit_lens_L7_mlp.png"),
                              img("advanced_logit_lens_L4_resid.png"),
                              img("advanced_logit_lens_L7_resid.png")],
                             ["L4/mlp", "L7/mlp", "L4/resid", "L7/resid"])

        # ── 7. CAUSAL ZERO-ABLATION ──
        ca_lines = [
            "Zero each syntax feature and measure the drop in P(keyword) at next-token.", "",
            "Positive drop = feature PROMOTES keyword. Negative = feature SUPPRESSES it.", "",
        ]
        for site_label in ["L7_resid", "L7_mlp", "L4_mlp"]:
            data = adv2.get(f"causal_ablation_{site_label}", {})
            if not data:
                continue
            ca_lines.append(f"  {site_label}:")
            for node in sorted(data, key=lambda n: -data[n].get("relative_drop", 0)):
                d = data[node]
                ca_lines.append(f"    {node:20s}  P_base={d['mean_baseline_prob']:.5f}  "
                                f"P_abl={d['mean_ablated_prob']:.5f}  "
                                f"rel_drop={d['relative_drop']:+.1%}")
            ca_lines.append("")

        ca_lines.extend([
            "  Key findings:",
            "  - L7/resid features are CAUSALLY NECESSARY: With (-74%), Yield (-67%), Assert (-51%)",
            "  - L7/mlp: Try (-41%), For (-19%) \u2014 moderate causal effects",
            "  - L4/resid ablation often INCREASES P(keyword) \u2014 suppressive/routing role",
        ])
        add_text_page(pdf, "7. Causal Zero-Ablation: Feature Necessity", ca_lines)

        add_image_page(pdf, "Causal Ablation: L7 (Causally Strongest)",
                        [img("advanced2_causal_ablation_L7_resid.png"),
                         img("advanced2_causal_ablation_L7_mlp.png")],
                        "L7/resid (left): large drops confirm causal necessity. L7/mlp (right): moderate effects.")

        # ── 8. COMPOSITIONAL NESTING ──
        nest_lines = [
            "When AST constructs are nested (e.g., 'for' inside 'if'), do both features fire?", "",
            "COMPOSE = both fire with >30% of standalone activation. INTERFERE = one suppresses other.", "",
            "  Case               L4/mlp     L4/resid    L7/mlp     L7/resid",
            "  " + "-" * 65,
        ]
        cases = ["for_in_if", "def_in_class", "if_in_for", "try_in_for", "with_in_def", "lambda_in_for"]
        for case in cases:
            parts = []
            for lbl in ["L4_mlp", "L4_resid", "L7_mlp", "L7_resid"]:
                c = adv2.get(f"nesting_{lbl}", {}).get(case, {})
                status = "COMPOSE" if c.get("composition_holds") else "INTERF." if c else "  N/A  "
                parts.append(f"{status:>8s}")
            nest_lines.append(f"  {case:20s}  {'  '.join(parts)}")

        compose_counts = {}
        for lbl in ["L4_mlp", "L4_resid", "L7_mlp", "L7_resid"]:
            d = adv2.get(f"nesting_{lbl}", {})
            compose_counts[lbl] = sum(1 for c in d.values() if c.get("composition_holds"))

        nest_lines.extend(["",
            f"  Compose rates: " + ", ".join(f"{k}: {v}/6" for k, v in compose_counts.items()), "",
            "  Features generally COMPOSE under nesting. The model maintains parallel",
            "  syntax representations. Exception: 'try_in_for' consistently fails.",
        ])
        add_text_page(pdf, "8. Compositional Nesting: Feature Composition", nest_lines)

        add_quad_image_page(pdf, "Compositional Nesting Visualizations",
                             [img("advanced2_nesting_L4_mlp.png"),
                              img("advanced2_nesting_L7_resid.png"),
                              img("advanced2_nesting_L7_mlp.png"),
                              img("advanced2_nesting_L4_resid.png")],
                             ["L4/mlp (5/6)", "L7/resid (5/6)", "L7/mlp (3/6)", "L4/resid (4/6)"])

        # ── 9. LINEAR PROBE ──
        lp_lines = [
            "Linear probe (logistic regression) on activations to classify 13 AST node types.",
            "Comparison: raw activations vs SAE latent codes. Chance = 7.7%.", "",
            "  Site         Raw Acc    SAE Acc    Gap       F1 (raw / SAE)",
            "  " + "-" * 60,
        ]
        for lbl in ["L4_mlp", "L4_resid", "L7_mlp", "L7_resid"]:
            d = adv2.get(f"linear_probe_{lbl}", {})
            if "error" in d:
                continue
            raw = d["raw_activation_probe"]
            sae = d["sae_latent_probe"]
            gap = raw["accuracy"] - sae["accuracy"]
            lp_lines.append(f"  {lbl:12s}  {raw['accuracy']:.1%}       {sae['accuracy']:.1%}      "
                            f"{gap:+.1%}      {raw['weighted_f1']:.3f} / {sae['weighted_f1']:.3f}")

        lp_lines.extend(["", "",
            "  Residual stream SAE latents preserve nearly ALL syntax information:",
            "    L7/resid: 72.8% vs 74.0% (only 1.2pp loss from sparse decomposition)",
            "",
            "  The TopK SAE is a FAITHFUL factorization of syntax information.",
        ])
        add_text_page(pdf, "9. Linear Probe: SAE Faithfulness Validation", lp_lines)

        add_image_page(pdf, "Linear Probe Results",
                        [img("advanced2_linear_probe.png")],
                        "Left: Overall accuracy. Right: Per-class accuracy for best site.")

        # ── 10. MINIMAL PAIRS ──
        mp_lines = [
            "Minimal code pairs differing in one AST construct.", "",
            "  Site         Accuracy",
            "  " + "-" * 30,
        ]
        for lbl in ["L4_mlp", "L4_resid", "L7_mlp", "L7_resid"]:
            d = adv2.get(f"minimal_pairs_{lbl}", {})
            correct = total = 0
            for pair in d.values():
                for diff in pair.get("differentials", {}).values():
                    total += 1
                    if diff.get("correct"):
                        correct += 1
            mp_lines.append(f"  {lbl:12s}   {correct}/{total} = {correct/max(total,1):.0%}")

        mp_lines.extend(["", "",
            "  L7/resid achieves 69% on controlled pairs.",
            "  L7/mlp is worst (25%): features capture shared context, not keywords.",
        ])
        add_text_page(pdf, "10. Minimal Pairs: Controlled Syntax Discrimination", mp_lines)

        add_image_page(pdf, "Minimal Pairs: Best Site (L7/resid)",
                        [img("advanced2_minimal_pairs_L7_resid.png")],
                        "Feature activation on code A vs B. Check/cross = correct/incorrect discrimination.")

        # ── 11. STEERING (NEGATIVE RESULT) ──
        add_text_page(pdf, "11. Feature Steering: Negative Result", [
            "Amplifying SAE features during generation to induce target syntax.", "",
            "Method: Multiply feature activation by 2x, 5x, 10x during generation.",
            "Measure: Does the keyword appear more often in steered output?", "",
            "Result: Model degenerates into '!!!!!...' under steering. Keyword rates ~0%.", "",
            "Short steering (5 tokens) shows occasional hits:",
            "  - For at 5x in L7/mlp, ListComp at L4/mlp, FunctionDef at 2x L4/mlp", "",
            "Interpretation:",
            "  These features are DETECTORS, not GENERATORS. They activate in response",
            "  to syntax but don't control next-token production. Logit lens confirms:",
            "  best features only promote keywords at rank 3-7, never rank 1.", "",
            "  This constrains the mechanistic role of these features to",
            "  detection/representation rather than generation/control.",
        ])

        # ── SUMMARY ──
        add_text_page(pdf, "Summary of Key Findings", [
            "",
            "1. HOURGLASS: Syntax specialization peaks mid-network (L4-L6), not output.",
            "",
            "2. DISJOINT SUBSPACES: MLP and resid features for the same AST node are",
            "   nearly orthogonal (cosine < 0.1). Complementary channels.",
            "",
            "3. RESIDUAL PERSISTENCE: Some features persist across 2-3 adjacent layers",
            "   in resid (ClassDef: cos 0.62), but MLP features are layer-independent.",
            "",
            "4. SEMANTIC CLUSTERS: Co-activation reveals expression-level and block-level",
            "   feature groupings in MLP.",
            "",
            "5. CAUSAL NECESSITY: L7/resid features are causally necessary for keyword",
            "   prediction (With: -74%, Yield: -67%, Assert: -51%).",
            "",
            "6. FAITHFUL DECOMPOSITION: SAE latent probes match raw probes within 1.2pp",
            "   at L7/resid (72.8% vs 74.0% on 13-class classification).",
            "",
            "7. COMPOSITION: Features compose under nesting \u2014 both outer and inner fire",
            "   simultaneously in 5/6 test cases.",
            "",
            "8. DETECTION NOT GENERATION: Feature steering fails \u2014 features detect syntax",
            "   but don't control generation. Important negative result.",
            "",
            "Overarching narrative: The model distributes syntax processing across sites",
            "and layers in a structured, non-redundant way. SAE features faithfully",
            "decompose this distributed representation, but function as detectors of",
            "syntactic context rather than generators of syntax tokens.",
        ], fontsize=10)

    print(f"Report saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
