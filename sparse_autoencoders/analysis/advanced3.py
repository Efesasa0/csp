#!/usr/bin/env python3
"""
Advanced Analysis Suite 3 — Eight additional mechanistic interpretability analyses.

1. Monosemanticity scoring:     activation entropy across AST node types per feature
2. Feature sufficiency:         reconstruct from top-k selective features only
3. Attention pattern analysis:  which attention heads correlate with feature activations
4. Feature frequency vs importance: scatter of firing rate vs causal ablation effect
5. Dead feature analysis:       fraction of SAE latents that never fire
6. Style robustness:            features robust to superficial code changes?
7. Residual→MLP information flow: what AST info does MLP add within a layer?
8. Feature superposition:       interference between decoder directions

Loads model once. Caches everything.
"""

import os, sys, json, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (
    load_config, get_device, get_layers, get_n_embd,
    resolve_path, data_path, ensure_dirs, patch_environment,
    get_sae_path, load_sae_from_checkpoint,
)

LAYERS = [4, 5, 6, 7]
SITES  = ["mlp", "resid"]

NODE_KEYWORDS = {
    "For": "for", "While": "while", "If": "if", "FunctionDef": "def",
    "AsyncFunctionDef": "async", "ClassDef": "class", "With": "with",
    "Try": "try", "Lambda": "lambda", "Return": "return", "Yield": "yield",
    "ListComp": "[", "Assert": "assert",
}


def load_best_features(cfg):
    path = data_path("results/multilayer_multisite_discovery.json")
    with open(path) as f:
        data = json.load(f)
    best = {}
    for key, layer_data in data.items():
        nodes = layer_data.get("nodes", {})
        best[key] = {}
        for node, info in nodes.items():
            top_sel = info.get("top_by_selectivity", [])
            winners = info.get("winner_features", [])
            if top_sel and top_sel[0].get("feat") is not None and top_sel[0]["sel"] > 0:
                best[key][node] = top_sel[0]["feat"]
            elif winners:
                best[key][node] = winners[0]
    return best


def get_hook_target(model, layer, site):
    layers = get_layers(model)
    return layers[layer].mlp if site == "mlp" else layers[layer]


def _harvest_latents(model, tokenizer, sae, device, layer, site, prompts,
                     max_len=256, last_token_only=False):
    """Run prompts through model+SAE, return latent activations.

    Returns list of tensors: [seq_len, n_hidden] per prompt (or [n_hidden] if last_token_only).
    """
    target = get_hook_target(model, layer, site)
    all_latents = []
    for code in prompts:
        enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 3:
            continue

        acts_store = []
        def capture(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            acts_store.append(x.detach())

        h = target.register_forward_hook(capture)
        with torch.no_grad():
            model(input_ids=input_ids)
        h.remove()

        if not acts_store:
            continue
        activation = acts_store[0][0]  # [seq_len, d_model]
        _, latents = sae(activation)   # [seq_len, n_hidden]

        if last_token_only:
            all_latents.append(latents[-1].cpu())
        else:
            all_latents.append(latents.cpu())

    return all_latents


# ══════════════════════════════════════════════════════════════════════
#  1. MONOSEMANTICITY SCORING
# ══════════════════════════════════════════════════════════════════════

def monosemanticity_scoring(model, tokenizer, sae, device, layer, site,
                            best_features, df, target_nodes, n_per_node=50,
                            cache_dir=""):
    """Measure activation entropy of each best feature across AST node types.

    A perfectly monosemantic feature fires on exactly one node type (entropy=0).
    A polysemantic feature fires uniformly across node types (entropy=log(n_nodes)).
    """
    cache_path = os.path.join(cache_dir, f"monosemanticity_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] monosemanticity L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)
    all_feat_ids = list(best_features.values())
    feat_to_node = {v: k for k, v in best_features.items()}

    # Collect mean activation per feature per node type
    node_prompts = {}
    for node in target_nodes:
        rows = df[df["ast_node"] == node]["prompt_text"].tolist()
        random.seed(42)
        random.shuffle(rows)
        node_prompts[node] = rows[:n_per_node]

    # feat_id -> {node: mean_activation}
    feat_node_acts = {f: {} for f in all_feat_ids}

    for node, prompts in tqdm(node_prompts.items(), desc=f"  Mono L{layer}/{site}"):
        latents_list = _harvest_latents(
            model, tokenizer, sae, device, layer, site, prompts,
            last_token_only=True
        )
        if not latents_list:
            continue
        latent_matrix = torch.stack(latents_list)  # [n_prompts, n_hidden]

        for f_id in all_feat_ids:
            mean_act = latent_matrix[:, f_id].mean().item()
            feat_node_acts[f_id][node] = mean_act

    results = {}
    max_entropy = math.log(max(len(target_nodes), 1))

    for node, f_id in best_features.items():
        acts_by_node = feat_node_acts[f_id]
        if not acts_by_node:
            continue

        # Normalize to distribution
        vals = np.array([max(acts_by_node.get(n, 0), 0) for n in target_nodes])
        total = vals.sum()
        if total < 1e-8:
            results[node] = {
                "feature": f_id, "entropy": None, "monosemanticity": None,
                "activations_by_node": {n: 0.0 for n in target_nodes},
            }
            continue

        probs = vals / total
        # Shannon entropy
        entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)
        mono_score = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 1.0

        # Which node gets highest activation?
        top_node = target_nodes[int(np.argmax(vals))]
        top_frac = float(vals.max() / total)

        results[node] = {
            "feature": f_id,
            "entropy": round(entropy, 4),
            "max_entropy": round(max_entropy, 4),
            "monosemanticity": round(mono_score, 4),
            "top_node": top_node,
            "top_fraction": round(top_frac, 4),
            "assigned_correct": top_node == node,
            "activations_by_node": {n: round(float(v), 6) for n, v in zip(target_nodes, vals)},
        }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_monosemanticity(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    mono_data = {}
    for key in sorted(all_results):
        if key.startswith("monosemanticity_"):
            mono_data[key.replace("monosemanticity_", "")] = all_results[key]

    if not mono_data:
        return

    # Summary plot: monosemanticity scores across layers/sites
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: bar chart of mean monosemanticity per (layer, site)
    labels = sorted(mono_data.keys())
    means = []
    for label in labels:
        scores = [v["monosemanticity"] for v in mono_data[label].values()
                  if v.get("monosemanticity") is not None]
        means.append(np.mean(scores) if scores else 0)

    ax = axes[0]
    colors = ["#4C72B0" if "mlp" in l else "#DD8452" for l in labels]
    ax.bar(range(len(labels)), means, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean monosemanticity (1 - normalized entropy)")
    ax.set_title("Feature Monosemanticity by Layer/Site")
    ax.set_ylim(0, 1)
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    # Right: per-node monosemanticity for best combo
    best_label = max(mono_data.keys(),
                     key=lambda l: np.mean([v.get("monosemanticity", 0)
                                            for v in mono_data[l].values()
                                            if v.get("monosemanticity") is not None]))
    best = mono_data[best_label]
    nodes = sorted([n for n in best if best[n].get("monosemanticity") is not None])
    scores = [best[n]["monosemanticity"] for n in nodes]
    correct = [best[n].get("assigned_correct", False) for n in nodes]
    bar_colors = ["#55A868" if c else "#C44E52" for c in correct]

    ax = axes[1]
    ax.barh(range(len(nodes)), scores, color=bar_colors)
    ax.set_yticks(range(len(nodes)))
    ax.set_yticklabels(nodes, fontsize=8)
    ax.set_xlabel("Monosemanticity score")
    ax.set_title(f"Per-node monosemanticity ({best_label})\nGreen=top node matches assigned")
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "advanced3_monosemanticity.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  2. FEATURE SUFFICIENCY (top-k reconstruction)
# ══════════════════════════════════════════════════════════════════════

def feature_sufficiency(model, tokenizer, sae, device, layer, site,
                        best_features, df, target_nodes, n_prompts=100,
                        cache_dir=""):
    """Test whether top-k AST-selective features are *sufficient* for classification.

    Reconstruct activations using ONLY the discovered selective features,
    then train a linear probe. Compare vs full reconstruction and raw activations.
    """
    cache_path = os.path.join(cache_dir, f"sufficiency_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] sufficiency L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target_mod = get_hook_target(model, layer, site)
    selective_feat_ids = set(best_features.values())

    node_to_idx = {n: i for i, n in enumerate(target_nodes)}
    X_raw, X_full_recon, X_selective_recon, y_list = [], [], [], []

    for node in tqdm(target_nodes, desc=f"  Sufficiency L{layer}/{site}"):
        kw = NODE_KEYWORDS.get(node, node.lower())
        mask = df["prompt_text"].str.contains(kw, case=True, na=False, regex=False)
        matching = df[mask]["prompt_text"].tolist()
        random.seed(42)
        random.shuffle(matching)
        count = 0

        for code in matching:
            if count >= n_prompts:
                break
            enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
            input_ids = enc.input_ids.to(device)
            if input_ids.shape[1] < 3:
                continue

            acts_store = []
            def hook(m, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                acts_store.append(x.detach())

            h = target_mod.register_forward_hook(hook)
            with torch.no_grad():
                model(input_ids=input_ids)
            h.remove()

            if not acts_store:
                continue
            raw_act = acts_store[0][0, -1, :]  # [d_model]
            x_hat_full, z = sae(raw_act.unsqueeze(0))

            # Selective reconstruction: zero out non-selective features
            z_selective = z.clone()
            mask_tensor = torch.ones(z.shape[-1], dtype=torch.bool, device=device)
            for fid in selective_feat_ids:
                if fid < mask_tensor.shape[0]:
                    mask_tensor[fid] = False
            z_selective[..., mask_tensor] = 0.0
            x_hat_selective = sae.decoder(z_selective) + sae.b_dec

            X_raw.append(raw_act.cpu())
            X_full_recon.append(x_hat_full[0].cpu())
            X_selective_recon.append(x_hat_selective[0].cpu())
            y_list.append(node_to_idx[node])
            count += 1

    if len(X_raw) < 50:
        results = {"error": "insufficient data"}
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    X_raw_np = torch.stack(X_raw).detach().numpy()
    X_full_np = torch.stack(X_full_recon).detach().numpy()
    X_sel_np = torch.stack(X_selective_recon).detach().numpy()
    y = np.array(y_list)

    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    probe_results = {}
    for name, X in [("raw", X_raw_np), ("full_recon", X_full_np), ("selective_recon", X_sel_np)]:
        probe = LogisticRegression(max_iter=1000, C=1.0)
        probe.fit(X[train_idx], y[train_idx])
        y_pred = probe.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred, average="weighted")
        probe_results[name] = {
            "accuracy": round(float(acc), 4),
            "weighted_f1": round(float(f1), 4),
        }

    results = {
        "n_train": int(split),
        "n_test": int(n - split),
        "n_selective_features": len(selective_feat_ids),
        "n_total_features": int(X_full_np.shape[1]) if X_full_np.ndim > 1 else 0,
        "probes": probe_results,
        "sufficiency_gap": round(
            probe_results["raw"]["accuracy"] - probe_results["selective_recon"]["accuracy"], 4
        ),
    }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_sufficiency(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    suff_data = {}
    for key in sorted(all_results):
        if key.startswith("sufficiency_"):
            data = all_results[key]
            if "error" not in data:
                suff_data[key.replace("sufficiency_", "")] = data

    if not suff_data:
        return

    labels = sorted(suff_data.keys())
    raw_accs = [suff_data[l]["probes"]["raw"]["accuracy"] for l in labels]
    full_accs = [suff_data[l]["probes"]["full_recon"]["accuracy"] for l in labels]
    sel_accs = [suff_data[l]["probes"]["selective_recon"]["accuracy"] for l in labels]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, raw_accs, w, label="Raw activations", color="#4C72B0")
    ax.bar(x, full_accs, w, label="Full SAE recon", color="#DD8452")
    ax.bar(x + w, sel_accs, w, label="Selective features only", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Feature Sufficiency: Can selective features alone classify AST nodes?")
    ax.legend()
    ax.set_ylim(0, 1)

    for i in range(len(labels)):
        for j, (vals, offset) in enumerate([(raw_accs, -w), (full_accs, 0), (sel_accs, w)]):
            ax.text(i + offset, vals[i] + 0.02, f"{vals[i]:.2f}", ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "advanced3_sufficiency.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  3. ATTENTION PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def attention_pattern_analysis(model, tokenizer, sae, device, layer, site,
                               best_features, df, n_prompts=100, cache_dir=""):
    """For residual-stream features, measure which attention heads' outputs
    correlate most with feature activation magnitude.

    For MLP features, correlate with the MLP input (post-attention residual).
    """
    cache_path = os.path.join(cache_dir, f"attention_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] attention L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    model_layers = get_layers(model)

    # Get number of attention heads
    attn_module = model_layers[layer].attn
    n_heads = model.config.n_head if hasattr(model.config, "n_head") else model.config.num_attention_heads

    target = get_hook_target(model, layer, site)

    prompts = df["prompt_text"].dropna().tolist()
    random.seed(42)
    random.shuffle(prompts)
    prompts = prompts[:n_prompts]

    all_feat_ids = list(best_features.values())
    feat_to_node = {v: k for k, v in best_features.items()}

    # For each prompt, collect: attention weights per head + feature activations
    head_feat_corrs = {f: np.zeros(n_heads) for f in all_feat_ids}
    head_counts = np.zeros(n_heads)
    n_samples = 0

    for code in tqdm(prompts, desc=f"  Attention L{layer}/{site}"):
        enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 3:
            continue

        # Capture attention weights
        attn_weights_store = []
        def attn_hook(m, inp, out):
            # GPT-2 style: attention module outputs (attn_output, attn_weights)
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                attn_weights_store.append(out[1].detach())

        # Capture SAE latents
        acts_store = []
        def act_hook(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            acts_store.append(x.detach())

        # Register hooks
        ha = attn_module.register_forward_hook(attn_hook)
        ht = target.register_forward_hook(act_hook)

        with torch.no_grad():
            model(input_ids=input_ids, output_attentions=True)
        ha.remove()
        ht.remove()

        if not acts_store:
            continue
        activation = acts_store[0][0]  # [seq_len, d_model]
        _, latents = sae(activation)    # [seq_len, n_hidden]

        # Get attention weights: [n_heads, seq_len, seq_len]
        if attn_weights_store:
            attn_w = attn_weights_store[0][0]  # [n_heads, seq_len, seq_len]
        else:
            continue

        seq_len = latents.shape[0]
        if attn_w.shape[1] != seq_len:
            continue

        # For each feature, compute correlation between feature activation at each position
        # and the attention each head pays TO that position (mean over query positions)
        for f_id in all_feat_ids:
            feat_act = latents[:, f_id].cpu().numpy()  # [seq_len]
            if feat_act.max() < 1e-5:
                continue

            for h_idx in range(min(n_heads, attn_w.shape[0])):
                # Mean attention received per position from all queries
                attn_received = attn_w[h_idx].mean(dim=0).cpu().numpy()  # [seq_len]
                # Pearson correlation
                if np.std(feat_act) > 1e-8 and np.std(attn_received) > 1e-8:
                    corr = np.corrcoef(feat_act, attn_received)[0, 1]
                    if not np.isnan(corr):
                        head_feat_corrs[f_id][h_idx] += corr

        n_samples += 1

    if n_samples == 0:
        results = {"error": "no valid samples"}
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    # Average correlations
    results = {}
    for node, f_id in best_features.items():
        corrs = head_feat_corrs[f_id] / max(n_samples, 1)
        top_heads = sorted(range(n_heads), key=lambda h: -abs(corrs[h]))[:5]
        results[node] = {
            "feature": f_id,
            "head_correlations": [round(float(c), 4) for c in corrs],
            "top_correlated_heads": [
                {"head": h, "correlation": round(float(corrs[h]), 4)}
                for h in top_heads
            ],
            "max_abs_correlation": round(float(np.max(np.abs(corrs))), 4),
        }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_attention(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])

    attn_data = {}
    for key in sorted(all_results):
        if key.startswith("attention_"):
            data = all_results[key]
            if "error" not in data:
                attn_data[key.replace("attention_", "")] = data

    if not attn_data:
        return

    for label, data in attn_data.items():
        nodes = [n for n in sorted(data.keys()) if "head_correlations" in data[n]]
        if not nodes:
            continue

        # Build heatmap: nodes x heads
        n_heads = len(data[nodes[0]]["head_correlations"])
        matrix = np.zeros((len(nodes), n_heads))
        for i, node in enumerate(nodes):
            matrix[i] = data[node]["head_correlations"]

        fig, ax = plt.subplots(figsize=(max(10, n_heads * 0.6), max(6, len(nodes) * 0.5)))
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes, fontsize=8)
        ax.set_xticks(range(n_heads))
        ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=7, rotation=45)
        ax.set_xlabel("Attention Head")
        ax.set_ylabel("AST Node Feature")
        ax.set_title(f"Feature–Attention Head Correlation ({label})")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Pearson r")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced3_attention_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  4. FEATURE FREQUENCY vs IMPORTANCE
# ══════════════════════════════════════════════════════════════════════

def frequency_vs_importance(model, tokenizer, sae, device, layer, site,
                            best_features, df, n_prompts=200, cache_dir=""):
    """Scatter: how often a feature fires vs its causal ablation effect.

    Reuses causal_ablation results if available, otherwise computes a lightweight
    version.
    """
    cache_path = os.path.join(cache_dir, f"freq_importance_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] freq_importance L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)

    # Load existing ablation results if available
    ablation_cache = os.path.join(
        os.path.dirname(cache_dir.rstrip("/")),
        "advanced2",
        f"causal_ablation_L{layer}_{site}.json"
    )
    ablation_data = {}
    if os.path.exists(ablation_cache):
        with open(ablation_cache) as f:
            ablation_data = json.load(f)

    # Compute firing rates
    prompts = df["prompt_text"].dropna().tolist()
    random.seed(42)
    random.shuffle(prompts)
    prompts = prompts[:n_prompts]

    all_feat_ids = list(best_features.values())
    feat_fire_counts = {f: 0 for f in all_feat_ids}
    feat_total_positions = 0

    for code in tqdm(prompts, desc=f"  FreqImport L{layer}/{site}"):
        enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 3:
            continue

        acts_store = []
        def capture(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            acts_store.append(x.detach())

        h = target.register_forward_hook(capture)
        with torch.no_grad():
            model(input_ids=input_ids)
        h.remove()

        if not acts_store:
            continue
        activation = acts_store[0][0]
        _, latents = sae(activation)
        seq_len = latents.shape[0]
        feat_total_positions += seq_len

        for f_id in all_feat_ids:
            feat_fire_counts[f_id] += (latents[:, f_id] > 1e-5).sum().item()

    results = {}
    for node, f_id in best_features.items():
        firing_rate = feat_fire_counts[f_id] / max(feat_total_positions, 1)

        # Get causal importance from ablation data
        ablation_drop = None
        if node in ablation_data:
            ablation_drop = ablation_data[node].get("prob_drop", None)

        results[node] = {
            "feature": f_id,
            "firing_rate": round(firing_rate, 6),
            "n_firings": feat_fire_counts[f_id],
            "total_positions": feat_total_positions,
            "causal_prob_drop": round(float(ablation_drop), 6) if ablation_drop is not None else None,
        }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_freq_importance(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])

    fi_data = {}
    for key in sorted(all_results):
        if key.startswith("freq_importance_"):
            fi_data[key.replace("freq_importance_", "")] = all_results[key]

    if not fi_data:
        return

    # One scatter per (layer, site)
    n_plots = len(fi_data)
    ncols = min(4, n_plots)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for idx, (label, data) in enumerate(sorted(fi_data.items())):
        ax = axes_flat[idx]
        nodes = [n for n in data if data[n].get("causal_prob_drop") is not None]
        if not nodes:
            ax.set_title(f"{label} (no ablation data)")
            continue

        freqs = [data[n]["firing_rate"] for n in nodes]
        drops = [data[n]["causal_prob_drop"] for n in nodes]

        ax.scatter(freqs, drops, s=40, alpha=0.7, color="#4C72B0")
        for i, n in enumerate(nodes):
            ax.annotate(n, (freqs[i], drops[i]), fontsize=6, alpha=0.8)

        ax.set_xlabel("Firing rate", fontsize=8)
        ax.set_ylabel("Causal P(kw) drop", fontsize=8)
        ax.set_title(f"Freq vs Importance ({label})", fontsize=9)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.4)

    for idx in range(n_plots, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.suptitle("Feature Frequency vs Causal Importance", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "advanced3_freq_importance.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  5. DEAD FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def dead_feature_analysis(model, tokenizer, sae, device, layer, site,
                          df, n_prompts=500, cache_dir=""):
    """What fraction of SAE latents never fire (or fire on <1% of inputs)?

    Speaks to SAE training quality and capacity usage.
    """
    cache_path = os.path.join(cache_dir, f"dead_features_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] dead_features L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)
    n_hidden = sae.encoder.out_features

    prompts = df["prompt_text"].dropna().tolist()
    random.seed(42)
    random.shuffle(prompts)
    prompts = prompts[:n_prompts]

    # Track per-feature: number of prompts where it fires at any position
    fire_counts = torch.zeros(n_hidden)
    n_valid = 0

    for code in tqdm(prompts, desc=f"  DeadFeats L{layer}/{site}"):
        enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 3:
            continue

        acts_store = []
        def capture(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            acts_store.append(x.detach())

        h = target.register_forward_hook(capture)
        with torch.no_grad():
            model(input_ids=input_ids)
        h.remove()

        if not acts_store:
            continue
        activation = acts_store[0][0]
        _, latents = sae(activation)

        # Any position fires?
        fires_any = (latents > 1e-5).any(dim=0).cpu().float()  # [n_hidden]
        fire_counts += fires_any
        n_valid += 1

    if n_valid == 0:
        results = {"error": "no valid samples"}
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    fire_rates = fire_counts / n_valid

    # Categorize features
    dead = (fire_rates < 1e-6).sum().item()        # never fire
    rare = ((fire_rates >= 1e-6) & (fire_rates < 0.01)).sum().item()  # <1%
    moderate = ((fire_rates >= 0.01) & (fire_rates < 0.1)).sum().item()
    active = (fire_rates >= 0.1).sum().item()

    # Distribution stats
    nonzero_rates = fire_rates[fire_rates > 1e-6]

    results = {
        "n_total_features": int(n_hidden),
        "n_prompts_tested": n_valid,
        "dead_features": int(dead),
        "dead_fraction": round(dead / n_hidden, 4),
        "rare_features": int(rare),
        "rare_fraction": round(rare / n_hidden, 4),
        "moderate_features": int(moderate),
        "active_features": int(active),
        "active_fraction": round(active / n_hidden, 4),
        "mean_fire_rate": round(float(fire_rates.mean()), 6),
        "median_fire_rate": round(float(fire_rates.median()), 6),
        "fire_rate_percentiles": {
            "p10": round(float(fire_rates.quantile(0.1)), 6),
            "p25": round(float(fire_rates.quantile(0.25)), 6),
            "p50": round(float(fire_rates.quantile(0.5)), 6),
            "p75": round(float(fire_rates.quantile(0.75)), 6),
            "p90": round(float(fire_rates.quantile(0.9)), 6),
            "p99": round(float(fire_rates.quantile(0.99)), 6),
        },
    }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_dead_features(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])

    dead_data = {}
    for key in sorted(all_results):
        if key.startswith("dead_features_"):
            data = all_results[key]
            if "error" not in data:
                dead_data[key.replace("dead_features_", "")] = data

    if not dead_data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: stacked bar of dead/rare/moderate/active per (layer, site)
    labels = sorted(dead_data.keys())
    dead_fracs = [dead_data[l]["dead_fraction"] for l in labels]
    rare_fracs = [dead_data[l]["rare_fraction"] for l in labels]
    mod_fracs = [dead_data[l]["moderate_features"] / dead_data[l]["n_total_features"] for l in labels]
    act_fracs = [dead_data[l]["active_fraction"] for l in labels]

    x = np.arange(len(labels))
    ax = axes[0]
    ax.bar(x, dead_fracs, label="Dead (0%)", color="#C44E52")
    ax.bar(x, rare_fracs, bottom=dead_fracs, label="Rare (<1%)", color="#DD8452")
    bottom2 = [d + r for d, r in zip(dead_fracs, rare_fracs)]
    ax.bar(x, mod_fracs, bottom=bottom2, label="Moderate (1-10%)", color="#CCBB44")
    bottom3 = [b + m for b, m in zip(bottom2, mod_fracs)]
    ax.bar(x, act_fracs, bottom=bottom3, label="Active (>10%)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of features")
    ax.set_title("Feature Utilization by Layer/Site")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    # Right: total feature counts
    ax = axes[1]
    totals = [dead_data[l]["n_total_features"] for l in labels]
    deads = [dead_data[l]["dead_features"] for l in labels]
    actives = [dead_data[l]["active_features"] for l in labels]

    w = 0.25
    ax.bar(x - w, totals, w, label="Total", color="#4C72B0")
    ax.bar(x, deads, w, label="Dead", color="#C44E52")
    ax.bar(x + w, actives, w, label="Active (>10%)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Feature count")
    ax.set_title("Feature Counts: Total vs Dead vs Active")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "advanced3_dead_features.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  6. STYLE ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════

STYLE_TRANSFORMS = {
    "rename_vars": {
        "For": (
            "def f(items):\n    for x in items:\n        process(x)",
            "def func(elements):\n    for elem in elements:\n        process(elem)",
        ),
        "While": (
            "def f(n):\n    while n > 0:\n        n -= 1\n    return n",
            "def func(count):\n    while count > 0:\n        count -= 1\n    return count",
        ),
        "If": (
            "def f(x):\n    if x > 0:\n        return x\n    return 0",
            "def func(value):\n    if value > 0:\n        return value\n    return 0",
        ),
        "FunctionDef": (
            "def compute(a, b):\n    return a + b",
            "def calculate(x, y):\n    return x + y",
        ),
        "ClassDef": (
            "class Worker:\n    def run(self):\n        pass",
            "class Employee:\n    def execute(self):\n        pass",
        ),
    },
    "add_comments": {
        "For": (
            "def f(items):\n    for x in items:\n        process(x)",
            "# Process all items in a loop\ndef f(items):\n    for x in items:  # iterate\n        process(x)",
        ),
        "While": (
            "def f(n):\n    while n > 0:\n        n -= 1\n    return n",
            "# Countdown function\ndef f(n):\n    while n > 0:  # keep going\n        n -= 1\n    return n",
        ),
        "If": (
            "def f(x):\n    if x > 0:\n        return x\n    return 0",
            "# Clamp to non-negative\ndef f(x):\n    if x > 0:  # check sign\n        return x\n    return 0",
        ),
        "FunctionDef": (
            "def compute(a, b):\n    return a + b",
            "# Addition helper\ndef compute(a, b):\n    return a + b  # sum",
        ),
        "ClassDef": (
            "class Worker:\n    def run(self):\n        pass",
            "# Worker abstraction\nclass Worker:\n    def run(self):  # main entry\n        pass",
        ),
    },
    "whitespace": {
        "For": (
            "def f(items):\n    for x in items:\n        process(x)",
            "def f( items ):\n    for  x  in  items:\n        process( x )",
        ),
        "While": (
            "def f(n):\n    while n > 0:\n        n -= 1\n    return n",
            "def f( n ):\n    while  n > 0:\n        n  -=  1\n    return  n",
        ),
        "If": (
            "def f(x):\n    if x > 0:\n        return x\n    return 0",
            "def f( x ):\n    if  x > 0:\n        return  x\n    return  0",
        ),
        "FunctionDef": (
            "def compute(a, b):\n    return a + b",
            "def compute( a , b ):\n    return  a  +  b",
        ),
        "ClassDef": (
            "class Worker:\n    def run(self):\n        pass",
            "class  Worker:\n    def  run( self ):\n        pass",
        ),
    },
}


def style_robustness(model, tokenizer, sae, device, layer, site,
                     best_features, cache_dir=""):
    """Test whether features are robust to superficial code changes
    (variable renaming, comments, whitespace) vs sensitive to structural changes.
    """
    cache_path = os.path.join(cache_dir, f"style_robust_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] style_robust L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)

    def get_max_acts(code):
        enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc.input_ids.to(device)
        acts_store = []
        def hook(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            acts_store.append(x.detach())
        h = target.register_forward_hook(hook)
        with torch.no_grad():
            model(input_ids=input_ids)
        h.remove()
        if not acts_store:
            return {}
        activation = acts_store[0][0]
        _, latents = sae(activation)
        result = {}
        for node, f_id in best_features.items():
            result[node] = latents[:, f_id].max().item()
        return result

    results = {}
    for transform_name, pairs in tqdm(STYLE_TRANSFORMS.items(), desc=f"  Style L{layer}/{site}"):
        transform_results = {}
        for node, (original, modified) in pairs.items():
            if node not in best_features:
                continue
            orig_acts = get_max_acts(original)
            mod_acts = get_max_acts(modified)

            orig_val = orig_acts.get(node, 0)
            mod_val = mod_acts.get(node, 0)

            # Robustness: how much does the feature activation change?
            if orig_val > 1e-5:
                relative_change = abs(mod_val - orig_val) / orig_val
            else:
                relative_change = float("inf") if mod_val > 1e-5 else 0.0

            transform_results[node] = {
                "original_activation": round(orig_val, 4),
                "modified_activation": round(mod_val, 4),
                "absolute_change": round(abs(mod_val - orig_val), 4),
                "relative_change": round(relative_change, 4) if relative_change != float("inf") else None,
                "robust": relative_change < 0.5,  # <50% change = robust
            }

        results[transform_name] = transform_results

    # Summary: per-node robustness across all transforms
    summary = {}
    for node in best_features:
        robust_count = 0
        total_count = 0
        for transform_name, transform_results in results.items():
            if node in transform_results:
                total_count += 1
                if transform_results[node]["robust"]:
                    robust_count += 1
        if total_count > 0:
            summary[node] = {
                "robust_transforms": robust_count,
                "total_transforms": total_count,
                "robustness_score": round(robust_count / total_count, 4),
            }
    results["_summary"] = summary

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_style_robustness(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])

    style_data = {}
    for key in sorted(all_results):
        if key.startswith("style_robust_"):
            style_data[key.replace("style_robust_", "")] = all_results[key]

    if not style_data:
        return

    # Summary heatmap: node x transform type, colored by robustness
    for label, data in style_data.items():
        transforms = [t for t in data if not t.startswith("_")]
        summary = data.get("_summary", {})
        nodes = sorted(summary.keys())
        if not nodes or not transforms:
            continue

        matrix = np.zeros((len(nodes), len(transforms)))
        for i, node in enumerate(nodes):
            for j, t in enumerate(transforms):
                if node in data.get(t, {}):
                    rel = data[t][node].get("relative_change")
                    matrix[i, j] = rel if rel is not None else 1.0

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: heatmap
        ax = axes[0]
        im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes, fontsize=8)
        ax.set_xticks(range(len(transforms)))
        ax.set_xticklabels([t.replace("_", "\n") for t in transforms], fontsize=8)
        ax.set_title(f"Relative activation change ({label})\nLower = more robust")
        plt.colorbar(im, ax=ax, shrink=0.7)

        # Right: overall robustness score per node
        ax = axes[1]
        rob_scores = [summary[n]["robustness_score"] for n in nodes]
        colors = ["#55A868" if s >= 0.5 else "#C44E52" for s in rob_scores]
        ax.barh(range(len(nodes)), rob_scores, color=colors)
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes, fontsize=8)
        ax.set_xlabel("Robustness score (fraction of transforms survived)")
        ax.set_title(f"Overall style robustness ({label})")
        ax.set_xlim(0, 1)
        ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced3_style_robust_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  7. RESIDUAL→MLP INFORMATION FLOW
# ══════════════════════════════════════════════════════════════════════

def residual_mlp_flow(model, tokenizer, saes, device, layer,
                      best_features, df, target_nodes, n_prompts=100,
                      cache_dir=""):
    """Compare residual-stream features (pre-MLP) vs MLP output features
    within the same layer. What AST information does the MLP sub-layer add?

    Requires both MLP and residual SAEs for the same layer.
    """
    cache_path = os.path.join(cache_dir, f"resid_mlp_flow_L{layer}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] resid_mlp_flow L{layer}")
        with open(cache_path) as f:
            return json.load(f)

    sae_resid = saes.get((layer, "resid"))
    sae_mlp = saes.get((layer, "mlp"))
    if sae_resid is None or sae_mlp is None:
        return {"error": f"missing SAE for layer {layer}"}

    model_layers = get_layers(model)
    resid_target = model_layers[layer]       # residual stream
    mlp_target = model_layers[layer].mlp     # MLP output

    bf_resid = best_features.get(f"L{layer}_resid", {})
    bf_mlp = best_features.get(f"L{layer}_mlp", {})

    prompts_by_node = {}
    for node in target_nodes:
        kw = NODE_KEYWORDS.get(node, node.lower())
        mask = df["prompt_text"].str.contains(kw, case=True, na=False, regex=False)
        matching = df[mask]["prompt_text"].tolist()
        random.seed(42)
        random.shuffle(matching)
        prompts_by_node[node] = matching[:n_prompts]

    results = {}
    for node in tqdm(target_nodes, desc=f"  ResidMLP L{layer}"):
        prompts = prompts_by_node.get(node, [])
        if not prompts:
            continue

        resid_acts_list = []
        mlp_acts_list = []

        for code in prompts:
            enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
            input_ids = enc.input_ids.to(device)
            if input_ids.shape[1] < 3:
                continue

            resid_store = []
            mlp_store = []

            def resid_hook(m, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                resid_store.append(x.detach())

            def mlp_hook(m, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                mlp_store.append(x.detach())

            hr = resid_target.register_forward_hook(resid_hook)
            hm = mlp_target.register_forward_hook(mlp_hook)
            with torch.no_grad():
                model(input_ids=input_ids)
            hr.remove()
            hm.remove()

            if resid_store and mlp_store:
                r_act = resid_store[0][0, -1, :]
                m_act = mlp_store[0][0, -1, :]

                _, z_resid = sae_resid(r_act.unsqueeze(0))
                _, z_mlp = sae_mlp(m_act.unsqueeze(0))

                resid_acts_list.append(z_resid[0].cpu())
                mlp_acts_list.append(z_mlp[0].cpu())

        if not resid_acts_list:
            continue

        resid_matrix = torch.stack(resid_acts_list)  # [n, n_hidden]
        mlp_matrix = torch.stack(mlp_acts_list)

        # Mean activation of the node's best feature
        resid_fid = bf_resid.get(node)
        mlp_fid = bf_mlp.get(node)

        node_result = {}
        if resid_fid is not None:
            node_result["resid_mean_act"] = round(resid_matrix[:, resid_fid].mean().item(), 6)
            node_result["resid_firing_rate"] = round(
                (resid_matrix[:, resid_fid] > 1e-5).float().mean().item(), 4
            )
        if mlp_fid is not None:
            node_result["mlp_mean_act"] = round(mlp_matrix[:, mlp_fid].mean().item(), 6)
            node_result["mlp_firing_rate"] = round(
                (mlp_matrix[:, mlp_fid] > 1e-5).float().mean().item(), 4
            )

        # Overall sparsity comparison
        node_result["resid_mean_sparsity"] = round(
            (resid_matrix > 1e-5).float().mean().item(), 4
        )
        node_result["mlp_mean_sparsity"] = round(
            (mlp_matrix > 1e-5).float().mean().item(), 4
        )

        # MLP adds information if mlp feature fires more than resid for this node
        if resid_fid is not None and mlp_fid is not None:
            node_result["mlp_gain"] = round(
                node_result["mlp_firing_rate"] - node_result["resid_firing_rate"], 4
            )

        results[node] = node_result

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_resid_mlp_flow(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])

    flow_data = {}
    for key in sorted(all_results):
        if key.startswith("resid_mlp_flow_"):
            data = all_results[key]
            if "error" not in data:
                flow_data[key.replace("resid_mlp_flow_", "")] = data

    if not flow_data:
        return

    for label, data in flow_data.items():
        nodes = [n for n in sorted(data.keys())
                 if "resid_firing_rate" in data[n] and "mlp_firing_rate" in data[n]]
        if not nodes:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: paired firing rates
        ax = axes[0]
        x = np.arange(len(nodes))
        w = 0.35
        resid_rates = [data[n]["resid_firing_rate"] for n in nodes]
        mlp_rates = [data[n]["mlp_firing_rate"] for n in nodes]
        ax.bar(x - w/2, resid_rates, w, label="Residual", color="#4C72B0")
        ax.bar(x + w/2, mlp_rates, w, label="MLP", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(nodes, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Firing rate of best feature")
        ax.set_title(f"Resid vs MLP feature firing ({label})")
        ax.legend()

        # Right: MLP gain
        ax = axes[1]
        gains = [data[n].get("mlp_gain", 0) for n in nodes]
        colors = ["#55A868" if g > 0 else "#C44E52" for g in gains]
        ax.barh(range(len(nodes)), gains, color=colors)
        ax.set_yticks(range(len(nodes)))
        ax.set_yticklabels(nodes, fontsize=8)
        ax.set_xlabel("MLP gain (MLP rate - Resid rate)")
        ax.set_title(f"MLP information gain ({label})")
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced3_resid_mlp_flow_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  8. FEATURE SUPERPOSITION MEASUREMENT
# ══════════════════════════════════════════════════════════════════════

def feature_superposition(sae, layer, site, best_features, cache_dir=""):
    """Measure superposition in the SAE dictionary.

    Metrics from Anthropic's "Toy Models of Superposition":
    - Mean pairwise cosine similarity of decoder directions
    - Interference score: for each feature, sum of squared cosine sims with all others
    - Fraction of decoder directions with high (>0.5) cosine sim to another direction
    """
    cache_path = os.path.join(cache_dir, f"superposition_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] superposition L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    W_dec = sae.decoder.weight.detach()  # [d_model, n_hidden]
    n_hidden = W_dec.shape[1]
    d_model = W_dec.shape[0]

    # Normalize decoder columns
    norms = W_dec.norm(dim=0, keepdim=True)  # [1, n_hidden]
    W_normed = W_dec / (norms + 1e-8)        # [d_model, n_hidden]

    # --- Global metrics (sample if too many features) ---
    max_sample = min(n_hidden, 4096)
    if n_hidden > max_sample:
        indices = torch.randperm(n_hidden)[:max_sample]
        W_sample = W_normed[:, indices]
    else:
        W_sample = W_normed
        indices = torch.arange(n_hidden)

    # Cosine similarity matrix for sampled features
    cos_matrix = (W_sample.T @ W_sample).cpu()  # [sample, sample]

    # Zero diagonal
    cos_matrix.fill_diagonal_(0)
    abs_cos = cos_matrix.abs()

    mean_abs_cos = abs_cos.mean().item()
    max_abs_cos = abs_cos.max().item()

    # Interference score per feature: sum of squared cosine sims
    interference = (cos_matrix ** 2).sum(dim=1)
    mean_interference = interference.mean().item()

    # Fraction with any high (>0.5) pairwise sim
    high_sim_mask = abs_cos > 0.5
    n_with_high_sim = high_sim_mask.any(dim=1).sum().item()
    frac_high_sim = n_with_high_sim / max_sample

    # --- Per-node metrics for best features ---
    node_metrics = {}
    all_feat_ids = list(best_features.values())
    valid_ids = [f for f in all_feat_ids if f < n_hidden]

    if valid_ids:
        # Get pairwise cosine for just the best features
        bf_cols = W_normed[:, valid_ids]  # [d_model, n_best]
        bf_cos = (bf_cols.T @ bf_cols).cpu()  # [n_best, n_best]
        bf_cos.fill_diagonal_(0)

        for i, (node, f_id) in enumerate(best_features.items()):
            if f_id >= n_hidden:
                continue
            idx_in_bf = valid_ids.index(f_id) if f_id in valid_ids else None
            if idx_in_bf is None:
                continue

            row = bf_cos[idx_in_bf]
            interference_local = (row ** 2).sum().item()
            max_sim_val = row.abs().max().item()
            max_sim_idx = row.abs().argmax().item()
            max_sim_node = list(best_features.keys())[max_sim_idx] if max_sim_idx < len(best_features) else "?"

            # Also check against ALL features
            feat_col = W_normed[:, f_id]  # [d_model]
            all_cos = (W_normed.T @ feat_col).cpu()  # [n_hidden]
            all_cos[f_id] = 0  # zero self
            global_max_sim = all_cos.abs().max().item()
            global_interference = (all_cos ** 2).sum().item()

            node_metrics[node] = {
                "feature": f_id,
                "decoder_norm": round(norms[0, f_id].item(), 4),
                "max_cos_sim_to_best_features": round(max_sim_val, 4),
                "most_similar_best_feature": max_sim_node,
                "interference_among_best": round(interference_local, 4),
                "global_max_cos_sim": round(global_max_sim, 4),
                "global_interference": round(global_interference, 4),
            }

    results = {
        "n_features": n_hidden,
        "d_model": d_model,
        "expansion_ratio": round(n_hidden / d_model, 1),
        "global_metrics": {
            "mean_abs_cosine": round(mean_abs_cos, 6),
            "max_abs_cosine": round(max_abs_cos, 4),
            "mean_interference": round(mean_interference, 4),
            "frac_high_sim_pairs": round(frac_high_sim, 4),
            "n_sampled": int(max_sample),
        },
        "per_node": node_metrics,
    }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_superposition(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])

    sup_data = {}
    for key in sorted(all_results):
        if key.startswith("superposition_"):
            sup_data[key.replace("superposition_", "")] = all_results[key]

    if not sup_data:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Left: global metrics comparison
    labels = sorted(sup_data.keys())
    ax = axes[0]
    mean_cos = [sup_data[l]["global_metrics"]["mean_abs_cosine"] for l in labels]
    mean_int = [sup_data[l]["global_metrics"]["mean_interference"] for l in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, mean_cos, 0.4, label="Mean |cos|", color="#4C72B0")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, mean_int, 0.4, label="Mean interference", color="#DD8452", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean |cosine similarity|")
    ax2.set_ylabel("Mean interference score")
    ax.set_title("Global Superposition Metrics")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)

    # Middle: fraction with high-sim pairs
    ax = axes[1]
    fracs = [sup_data[l]["global_metrics"]["frac_high_sim_pairs"] for l in labels]
    colors = ["#4C72B0" if "mlp" in l else "#DD8452" for l in labels]
    ax.bar(range(len(labels)), fracs, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of features")
    ax.set_title("Features with high (>0.5) cosine sim to another")
    ax.set_ylim(0, 1)

    # Right: per-node interference for best combo
    best_label = labels[0] if labels else None
    if best_label and "per_node" in sup_data[best_label]:
        ax = axes[2]
        per_node = sup_data[best_label]["per_node"]
        nodes = sorted(per_node.keys())
        global_ints = [per_node[n]["global_interference"] for n in nodes]
        norms = [per_node[n]["decoder_norm"] for n in nodes]

        ax.scatter(norms, global_ints, s=50, alpha=0.7, color="#4C72B0")
        for i, n in enumerate(nodes):
            ax.annotate(n, (norms[i], global_ints[i]), fontsize=7)
        ax.set_xlabel("Decoder norm")
        ax.set_ylabel("Global interference")
        ax.set_title(f"Decoder norm vs interference ({best_label})")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "advanced3_superposition.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    cfg = load_config()
    ensure_dirs(cfg)
    adv_cache = data_path("cache/advanced3")
    os.makedirs(adv_cache, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    patch_environment()
    device = get_device()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("openai/circuit-sparsity", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    ).to(device).eval()

    df = pd.read_parquet(resolve_path(cfg["paths"]["dataset"]))
    d_model = get_n_embd(model.config)
    target_nodes = cfg["sae_params"]["target_nodes"]

    best_features = load_best_features(cfg)

    # Load all SAEs
    print("Loading SAEs...")
    saes = {}
    for layer_idx in range(8):
        for site in SITES:
            sae_path = get_sae_path(cfg, layer_idx, site)
            if os.path.exists(sae_path):
                sae, meta = load_sae_from_checkpoint(sae_path, d_model, device)
                saes[(layer_idx, site)] = sae
                print(f"  L{layer_idx}/{site}: R²={meta.get('r2', '?')}")

    all_results = {}

    # ── 1. Monosemanticity Scoring ──
    print("\n=== Analysis 1: Monosemanticity Scoring ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"monosemanticity_{label}"] = monosemanticity_scoring(
                model, tokenizer, saes[key], device, layer, site, bf, df,
                target_nodes, cache_dir=adv_cache
            )

    # ── 2. Feature Sufficiency ──
    print("\n=== Analysis 2: Feature Sufficiency ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"sufficiency_{label}"] = feature_sufficiency(
                model, tokenizer, saes[key], device, layer, site, bf, df,
                target_nodes, cache_dir=adv_cache
            )

    # ── 3. Attention Pattern Analysis ──
    print("\n=== Analysis 3: Attention Pattern Analysis ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"attention_{label}"] = attention_pattern_analysis(
                model, tokenizer, saes[key], device, layer, site, bf, df,
                cache_dir=adv_cache
            )

    # ── 4. Frequency vs Importance ──
    print("\n=== Analysis 4: Frequency vs Importance ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"freq_importance_{label}"] = frequency_vs_importance(
                model, tokenizer, saes[key], device, layer, site, bf, df,
                cache_dir=adv_cache
            )

    # ── 5. Dead Feature Analysis ──
    print("\n=== Analysis 5: Dead Feature Analysis ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            all_results[f"dead_features_{label}"] = dead_feature_analysis(
                model, tokenizer, saes[key], device, layer, site, df,
                cache_dir=adv_cache
            )

    # ── 6. Style Robustness ──
    print("\n=== Analysis 6: Style Robustness ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"style_robust_{label}"] = style_robustness(
                model, tokenizer, saes[key], device, layer, site, bf,
                cache_dir=adv_cache
            )

    # ── 7. Residual→MLP Information Flow ──
    print("\n=== Analysis 7: Residual→MLP Information Flow ===")
    for layer in LAYERS:
        if (layer, "resid") in saes and (layer, "mlp") in saes:
            all_results[f"resid_mlp_flow_L{layer}"] = residual_mlp_flow(
                model, tokenizer, saes, device, layer,
                best_features, df, target_nodes, cache_dir=adv_cache
            )

    # ── 8. Feature Superposition ──
    print("\n=== Analysis 8: Feature Superposition ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"superposition_{label}"] = feature_superposition(
                saes[key], layer, site, bf, cache_dir=adv_cache
            )

    # ── Generate all visualizations ──
    print("\n=== Generating Visualizations ===")
    plot_monosemanticity(all_results, cfg)
    plot_sufficiency(all_results, cfg)
    plot_attention(all_results, cfg)
    plot_freq_importance(all_results, cfg)
    plot_dead_features(all_results, cfg)
    plot_style_robustness(all_results, cfg)
    plot_resid_mlp_flow(all_results, cfg)
    plot_superposition(all_results, cfg)

    # ── Save master results ──
    out_path = data_path("results/advanced_analysis_3.json")

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return round(float(obj), 6)
        return obj

    with open(out_path, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\nAll results saved to {out_path}")
    print(f"Visualizations saved to {resolve_path(cfg['paths']['dirs']['results'])}")


if __name__ == "__main__":
    main()
