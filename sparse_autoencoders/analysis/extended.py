#!/usr/bin/env python3
"""
Advanced Analysis Suite 2 — Five additional mechanistic interpretability analyses.

1. Causal zero-ablation: zero a feature → measure P(keyword) drop
2. Positional analysis:  WHERE in the token sequence do features fire?
3. Minimal pairs:        controlled code pairs differing in one AST node
4. Compositional nesting: do features compose when constructs are nested?
5. Linear probe:         baseline linear classifier on raw activations
"""

import os, sys, json, math, random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (
    load_config, get_device, get_layers, get_n_embd,
    resolve_path, data_path, ensure_dirs, patch_environment,
    get_sae_path, load_sae_from_checkpoint,
)

LAYERS = [4, 5, 6, 7]
SITES  = ["mlp", "resid"]

# ── Keyword map: AST node → token substring to search for ──
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


# ══════════════════════════════════════════════════════════════════════
#  1. CAUSAL ZERO-ABLATION
# ══════════════════════════════════════════════════════════════════════

def causal_ablation(model, tokenizer, sae, device, layer, site,
                    best_features, df, n_prompts=100, cache_dir=""):
    """Zero each feature and measure drop in P(keyword) at next-token position."""
    cache_path = os.path.join(cache_dir, f"causal_ablation_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] causal_ablation L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)
    results = {}

    # Gather prompts per node from the dataset
    node_prompts = {}
    for node in best_features:
        kw = NODE_KEYWORDS.get(node, node.lower())
        # Find prompts that contain this keyword
        mask = df["prompt_text"].str.contains(kw, case=True, na=False, regex=False)
        matching = df[mask]["prompt_text"].tolist()
        if matching:
            node_prompts[node] = matching[:n_prompts]

    for node, f_id in tqdm(best_features.items(), desc=f"  Ablation L{layer}/{site}"):
        kw = NODE_KEYWORDS.get(node, node.lower())
        prompts = node_prompts.get(node, [])
        if not prompts:
            continue

        # Get keyword token IDs
        kw_tokens = set()
        for t_str in [kw, " " + kw, kw.capitalize(), " " + kw.capitalize()]:
            ids = tokenizer.encode(t_str, add_special_tokens=False)
            kw_tokens.update(ids)
        if not kw_tokens:
            continue

        baseline_probs = []
        ablated_probs = []

        for code in prompts[:n_prompts]:
            input_ids = tokenizer(code, return_tensors="pt", truncation=True,
                                  max_length=512).input_ids.to(device)
            if input_ids.shape[1] < 5:
                continue

            # Baseline logits
            with torch.no_grad():
                base_logits = model(input_ids=input_ids).logits[0, -1]
            base_probs_all = torch.softmax(base_logits, dim=-1)
            base_kw_prob = sum(base_probs_all[t].item() for t in kw_tokens)

            # Ablated logits (zero out feature)
            def ablation_hook(m, inp, out, _sae=sae, _fid=f_id):
                x = out[0] if isinstance(out, tuple) else out
                _, z = _sae(x)
                z[..., _fid] = 0.0
                new_x = _sae.decoder(z) + _sae.b_dec
                return (new_x,) + out[1:] if isinstance(out, tuple) else new_x

            h = target.register_forward_hook(ablation_hook)
            with torch.no_grad():
                abl_logits = model(input_ids=input_ids).logits[0, -1]
            h.remove()
            abl_probs_all = torch.softmax(abl_logits, dim=-1)
            abl_kw_prob = sum(abl_probs_all[t].item() for t in kw_tokens)

            baseline_probs.append(base_kw_prob)
            ablated_probs.append(abl_kw_prob)

        if baseline_probs:
            mean_base = np.mean(baseline_probs)
            mean_abl = np.mean(ablated_probs)
            results[node] = {
                "feature": f_id,
                "mean_baseline_prob": round(float(mean_base), 6),
                "mean_ablated_prob": round(float(mean_abl), 6),
                "prob_drop": round(float(mean_base - mean_abl), 6),
                "relative_drop": round(float((mean_base - mean_abl) / max(mean_base, 1e-8)), 4),
                "n_prompts": len(baseline_probs),
            }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_causal_ablation(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    for key in sorted(all_results):
        if not key.startswith("causal_ablation_"):
            continue
        label = key.replace("causal_ablation_", "")
        data = all_results[key]
        if not data:
            continue

        nodes = sorted(data.keys())
        base_probs = [data[n]["mean_baseline_prob"] for n in nodes]
        abl_probs = [data[n]["mean_ablated_prob"] for n in nodes]
        drops = [data[n]["prob_drop"] for n in nodes]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: paired bar chart
        x = np.arange(len(nodes))
        w = 0.35
        axes[0].bar(x - w/2, base_probs, w, label="Baseline P(kw)", color="#4C72B0")
        axes[0].bar(x + w/2, abl_probs, w, label="Ablated P(kw)", color="#DD8452")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(nodes, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("P(keyword)")
        axes[0].set_title(f"Causal Ablation — P(keyword) ({label})")
        axes[0].legend(fontsize=8)

        # Right: relative drop
        rel_drops = [data[n]["relative_drop"] for n in nodes]
        colors = ["#C44E52" if d > 0 else "#55A868" for d in rel_drops]
        axes[1].barh(nodes, rel_drops, color=colors)
        axes[1].set_xlabel("Relative probability drop")
        axes[1].set_title(f"Relative P(keyword) drop ({label})")
        axes[1].axvline(0, color="black", linewidth=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced2_causal_ablation_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  2. POSITIONAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def positional_analysis(model, tokenizer, sae, device, layer, site,
                        best_features, df, n_prompts=200, cache_dir=""):
    """For each feature, record which token positions it fires at."""
    cache_path = os.path.join(cache_dir, f"positional_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] positional L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)
    results = {}

    samples = df["prompt_text"].dropna().tolist()
    random.seed(42)
    random.shuffle(samples)
    samples = samples[:n_prompts]

    all_feats = list(best_features.values())
    feat_to_node = {v: k for k, v in best_features.items()}

    # Collect per-feature: relative position of activations, token text at activation
    feat_positions = {f: [] for f in all_feats}
    feat_tokens = {f: [] for f in all_feats}

    for code in tqdm(samples, desc=f"  Positional L{layer}/{site}"):
        enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
        input_ids = enc.input_ids.to(device)
        seq_len = input_ids.shape[1]
        if seq_len < 5:
            continue

        acts_store = []
        def capture_hook(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            acts_store.append(x.detach())

        h = target.register_forward_hook(capture_hook)
        with torch.no_grad():
            model(input_ids=input_ids)
        h.remove()

        if not acts_store:
            continue
        activation = acts_store[0][0]  # [seq_len, d_model]

        # Run SAE on all positions
        _, latents = sae(activation)  # [seq_len, n_hidden]

        tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

        for f_id in all_feats:
            firing = (latents[:, f_id] > 1e-5).nonzero(as_tuple=True)[0]
            for pos in firing.tolist():
                rel_pos = pos / max(seq_len - 1, 1)
                feat_positions[f_id].append(rel_pos)
                if pos < len(tokens):
                    feat_tokens[f_id].append(tokens[pos].strip())

    for node, f_id in best_features.items():
        positions = feat_positions[f_id]
        token_strs = feat_tokens[f_id]
        if not positions:
            results[node] = {"feature": f_id, "n_firings": 0}
            continue

        # Token frequency at activation site
        from collections import Counter
        tok_counts = Counter(token_strs).most_common(10)

        results[node] = {
            "feature": f_id,
            "n_firings": len(positions),
            "mean_rel_position": round(float(np.mean(positions)), 4),
            "std_rel_position": round(float(np.std(positions)), 4),
            "median_rel_position": round(float(np.median(positions)), 4),
            "position_quartiles": [round(float(np.percentile(positions, q)), 4) for q in [25, 50, 75]],
            "top_tokens_at_fire": [{"token": t, "count": c} for t, c in tok_counts],
        }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_positional(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    for key in sorted(all_results):
        if not key.startswith("positional_"):
            continue
        label = key.replace("positional_", "")
        data = all_results[key]
        if not data:
            continue

        nodes = [n for n in sorted(data.keys()) if data[n].get("n_firings", 0) > 0]
        if not nodes:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: mean position with error bars
        means = [data[n]["mean_rel_position"] for n in nodes]
        stds = [data[n]["std_rel_position"] for n in nodes]
        y_pos = np.arange(len(nodes))
        axes[0].barh(y_pos, means, xerr=stds, color="#4C72B0", alpha=0.8, capsize=3)
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels(nodes, fontsize=8)
        axes[0].set_xlabel("Relative position in sequence (0=start, 1=end)")
        axes[0].set_title(f"Feature firing position ({label})")
        axes[0].set_xlim(0, 1)
        axes[0].axvline(0.5, color="gray", linestyle="--", alpha=0.5)

        # Right: top tokens at firing site (for top 6 nodes by n_firings)
        top_nodes = sorted(nodes, key=lambda n: -data[n]["n_firings"])[:6]
        cell_text = []
        for n in top_nodes:
            top_toks = data[n].get("top_tokens_at_fire", [])[:5]
            tok_str = ", ".join(f'"{t["token"]}"({t["count"]})' for t in top_toks)
            cell_text.append([n, tok_str])

        axes[1].axis("off")
        table = axes[1].table(cellText=cell_text, colLabels=["Node", "Top tokens at firing position"],
                              loc="center", cellLoc="left")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.5)
        axes[1].set_title(f"What tokens trigger features ({label})", fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced2_positional_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  3. MINIMAL PAIRS
# ══════════════════════════════════════════════════════════════════════

MINIMAL_PAIRS = [
    {
        "name": "For_vs_While",
        "a_node": "For", "b_node": "While",
        "a_code": "def f(items):\n    for x in items:\n        process(x)\n    return True",
        "b_code": "def f(items):\n    i = 0\n    while i < len(items):\n        process(items[i])\n        i += 1\n    return True",
    },
    {
        "name": "If_vs_While",
        "a_node": "If", "b_node": "While",
        "a_code": "def f(x):\n    if x > 0:\n        x = x - 1\n    return x",
        "b_code": "def f(x):\n    while x > 0:\n        x = x - 1\n    return x",
    },
    {
        "name": "FunctionDef_vs_Lambda",
        "a_node": "FunctionDef", "b_node": "Lambda",
        "a_code": "def square(x):\n    return x * x\nresult = square(5)",
        "b_code": "square = lambda x: x * x\nresult = square(5)",
    },
    {
        "name": "For_vs_ListComp",
        "a_node": "For", "b_node": "ListComp",
        "a_code": "result = []\nfor x in range(10):\n    result.append(x * 2)",
        "b_code": "result = [x * 2 for x in range(10)]",
    },
    {
        "name": "Try_vs_If",
        "a_node": "Try", "b_node": "If",
        "a_code": "def safe_div(a, b):\n    try:\n        return a / b\n    except:\n        return 0",
        "b_code": "def safe_div(a, b):\n    if b != 0:\n        return a / b\n    else:\n        return 0",
    },
    {
        "name": "With_vs_manual",
        "a_node": "With", "b_node": "FunctionDef",
        "a_code": "def read_file(path):\n    with open(path) as f:\n        data = f.read()\n    return data",
        "b_code": "def read_file(path):\n    f = open(path)\n    data = f.read()\n    f.close()\n    return data",
    },
    {
        "name": "Return_vs_Yield",
        "a_node": "Return", "b_node": "Yield",
        "a_code": "def get_items(data):\n    result = []\n    for x in data:\n        result.append(x)\n    return result",
        "b_code": "def get_items(data):\n    for x in data:\n        yield x",
    },
    {
        "name": "ClassDef_vs_FunctionDef",
        "a_node": "ClassDef", "b_node": "FunctionDef",
        "a_code": "class Counter:\n    def __init__(self):\n        self.count = 0\n    def increment(self):\n        self.count += 1",
        "b_code": "def make_counter():\n    count = 0\n    def increment():\n        nonlocal count\n        count += 1\n    return increment",
    },
]


def minimal_pairs(model, tokenizer, sae, device, layer, site,
                  best_features, cache_dir=""):
    """Compare feature activations on minimal code pairs."""
    cache_path = os.path.join(cache_dir, f"minimal_pairs_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] minimal_pairs L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)
    results = {}

    for pair in tqdm(MINIMAL_PAIRS, desc=f"  MinPairs L{layer}/{site}"):
        pair_results = {"a_node": pair["a_node"], "b_node": pair["b_node"]}

        for side in ["a", "b"]:
            code = pair[f"{side}_code"]
            enc = tokenizer(code, return_tensors="pt", truncation=True, max_length=256)
            input_ids = enc.input_ids.to(device)

            acts_store = []
            def capture_hook(m, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                acts_store.append(x.detach())

            h = target.register_forward_hook(capture_hook)
            with torch.no_grad():
                model(input_ids=input_ids)
            h.remove()

            if not acts_store:
                continue
            activation = acts_store[0][0]  # [seq_len, d_model]
            _, latents = sae(activation)  # [seq_len, n_hidden]

            # Max activation across all positions for each best feature
            feat_activations = {}
            for node, f_id in best_features.items():
                max_act = latents[:, f_id].max().item()
                mean_act = latents[:, f_id].mean().item()
                firing_frac = (latents[:, f_id] > 1e-5).float().mean().item()
                feat_activations[node] = {
                    "max_act": round(max_act, 4),
                    "mean_act": round(mean_act, 4),
                    "firing_frac": round(firing_frac, 4),
                }
            pair_results[f"{side}_activations"] = feat_activations

        # Compute differential: does the expected feature fire more on its own code?
        a_node, b_node = pair["a_node"], pair["b_node"]
        diffs = {}
        for node in [a_node, b_node]:
            if node not in best_features:
                continue
            a_act = pair_results.get("a_activations", {}).get(node, {}).get("max_act", 0)
            b_act = pair_results.get("b_activations", {}).get(node, {}).get("max_act", 0)
            # For a_node, we expect higher activation on a_code
            expected_higher = "a" if node == a_node else "b"
            actual_higher = "a" if a_act > b_act else ("b" if b_act > a_act else "tie")
            diffs[node] = {
                "a_max_act": round(a_act, 4),
                "b_max_act": round(b_act, 4),
                "expected_higher": expected_higher,
                "actual_higher": actual_higher,
                "correct": expected_higher == actual_higher,
                "margin": round(abs(a_act - b_act), 4),
            }
        pair_results["differentials"] = diffs
        results[pair["name"]] = pair_results

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_minimal_pairs(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    for key in sorted(all_results):
        if not key.startswith("minimal_pairs_"):
            continue
        label = key.replace("minimal_pairs_", "")
        data = all_results[key]
        if not data:
            continue

        pair_names = list(data.keys())
        n_pairs = len(pair_names)

        fig, axes = plt.subplots(2, (n_pairs + 1) // 2, figsize=(16, 8))
        axes = axes.flatten()

        for idx, pair_name in enumerate(pair_names):
            if idx >= len(axes):
                break
            ax = axes[idx]
            pair = data[pair_name]
            diffs = pair.get("differentials", {})

            nodes = list(diffs.keys())
            a_acts = [diffs[n]["a_max_act"] for n in nodes]
            b_acts = [diffs[n]["b_max_act"] for n in nodes]
            correct = [diffs[n]["correct"] for n in nodes]

            x = np.arange(len(nodes))
            w = 0.35
            bars_a = ax.bar(x - w/2, a_acts, w, label=f"Code A ({pair['a_node']})",
                           color="#4C72B0")
            bars_b = ax.bar(x + w/2, b_acts, w, label=f"Code B ({pair['b_node']})",
                           color="#DD8452")

            # Mark correct/incorrect
            for i, c in enumerate(correct):
                marker = "✓" if c else "✗"
                y_max = max(a_acts[i], b_acts[i])
                ax.text(i, y_max * 1.05, marker, ha="center", fontsize=10,
                        color="green" if c else "red")

            ax.set_xticks(x)
            ax.set_xticklabels(nodes, fontsize=7, rotation=30)
            ax.set_title(pair_name.replace("_", " "), fontsize=9)
            ax.legend(fontsize=6)

        for idx in range(n_pairs, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Minimal Pairs — Feature activation differentials ({label})", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced2_minimal_pairs_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  4. COMPOSITIONAL NESTING
# ══════════════════════════════════════════════════════════════════════

NESTING_CASES = [
    {
        "name": "for_in_if",
        "outer": "If", "inner": "For",
        "nested_code": "def f(data, flag):\n    if flag:\n        for x in data:\n            process(x)",
        "outer_only": "def f(data, flag):\n    if flag:\n        process(data)",
        "inner_only": "def f(data):\n    for x in data:\n        process(x)",
    },
    {
        "name": "def_in_class",
        "outer": "ClassDef", "inner": "FunctionDef",
        "nested_code": "class MyClass:\n    def method(self):\n        return self.value",
        "outer_only": "class MyClass:\n    pass",
        "inner_only": "def method(self):\n    return self.value",
    },
    {
        "name": "if_in_for",
        "outer": "For", "inner": "If",
        "nested_code": "def f(data):\n    for x in data:\n        if x > 0:\n            process(x)",
        "outer_only": "def f(data):\n    for x in data:\n        process(x)",
        "inner_only": "def f(x):\n    if x > 0:\n        process(x)",
    },
    {
        "name": "try_in_for",
        "outer": "For", "inner": "Try",
        "nested_code": "def f(items):\n    for x in items:\n        try:\n            process(x)\n        except:\n            pass",
        "outer_only": "def f(items):\n    for x in items:\n        process(x)",
        "inner_only": "def f(x):\n    try:\n        process(x)\n    except:\n        pass",
    },
    {
        "name": "with_in_def",
        "outer": "FunctionDef", "inner": "With",
        "nested_code": "def read(path):\n    with open(path) as f:\n        return f.read()",
        "outer_only": "def read(path):\n    return path",
        "inner_only": "with open('file.txt') as f:\n    data = f.read()",
    },
    {
        "name": "lambda_in_for",
        "outer": "For", "inner": "Lambda",
        "nested_code": "result = []\nfor x in range(10):\n    fn = lambda y: y * x\n    result.append(fn(x))",
        "outer_only": "result = []\nfor x in range(10):\n    result.append(x * x)",
        "inner_only": "fn = lambda x: x * 2\nresult = fn(5)",
    },
]


def compositional_nesting(model, tokenizer, sae, device, layer, site,
                          best_features, cache_dir=""):
    """Test whether features compose when AST constructs are nested."""
    cache_path = os.path.join(cache_dir, f"nesting_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] nesting L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)
    results = {}

    def get_feature_acts(code):
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
        feat_acts = {}
        for node, f_id in best_features.items():
            feat_acts[node] = {
                "max": round(latents[:, f_id].max().item(), 4),
                "mean": round(latents[:, f_id].mean().item(), 4),
                "firing_frac": round((latents[:, f_id] > 1e-5).float().mean().item(), 4),
            }
        return feat_acts

    for case in tqdm(NESTING_CASES, desc=f"  Nesting L{layer}/{site}"):
        outer, inner = case["outer"], case["inner"]
        if outer not in best_features or inner not in best_features:
            continue

        nested_acts = get_feature_acts(case["nested_code"])
        outer_acts = get_feature_acts(case["outer_only"])
        inner_acts = get_feature_acts(case["inner_only"])

        # Key question: in nested code, do both outer and inner features fire?
        outer_in_nested = nested_acts.get(outer, {}).get("max", 0)
        inner_in_nested = nested_acts.get(inner, {}).get("max", 0)
        outer_alone = outer_acts.get(outer, {}).get("max", 0)
        inner_alone = inner_acts.get(inner, {}).get("max", 0)

        results[case["name"]] = {
            "outer_node": outer,
            "inner_node": inner,
            "nested": {
                "outer_feat_max": outer_in_nested,
                "inner_feat_max": inner_in_nested,
                "both_fire": outer_in_nested > 1e-3 and inner_in_nested > 1e-3,
            },
            "outer_only": {
                "outer_feat_max": outer_alone,
                "inner_feat_max": outer_acts.get(inner, {}).get("max", 0),
            },
            "inner_only": {
                "outer_feat_max": inner_acts.get(outer, {}).get("max", 0),
                "inner_feat_max": inner_alone,
            },
            "composition_holds": (
                outer_in_nested > 1e-3 and inner_in_nested > 1e-3
                and outer_in_nested >= outer_alone * 0.3
                and inner_in_nested >= inner_alone * 0.3
            ),
        }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_nesting(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    for key in sorted(all_results):
        if not key.startswith("nesting_"):
            continue
        label = key.replace("nesting_", "")
        data = all_results[key]
        if not data:
            continue

        cases = list(data.keys())
        n_cases = len(cases)
        if n_cases == 0:
            continue

        fig, axes = plt.subplots(2, (n_cases + 1) // 2, figsize=(16, 8))
        axes = axes.flatten()

        for idx, case_name in enumerate(cases):
            if idx >= len(axes):
                break
            ax = axes[idx]
            c = data[case_name]
            outer, inner = c["outer_node"], c["inner_node"]

            categories = ["Nested", f"{outer}\nonly", f"{inner}\nonly"]
            outer_vals = [
                c["nested"]["outer_feat_max"],
                c["outer_only"]["outer_feat_max"],
                c["inner_only"]["outer_feat_max"],
            ]
            inner_vals = [
                c["nested"]["inner_feat_max"],
                c["outer_only"]["inner_feat_max"],
                c["inner_only"]["inner_feat_max"],
            ]

            x = np.arange(len(categories))
            w = 0.35
            ax.bar(x - w/2, outer_vals, w, label=f"{outer} feat", color="#4C72B0")
            ax.bar(x + w/2, inner_vals, w, label=f"{inner} feat", color="#DD8452")
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=7)

            status = "COMPOSE" if c["composition_holds"] else "INTERFERE"
            color = "green" if c["composition_holds"] else "red"
            ax.set_title(f"{case_name} [{status}]", fontsize=9, color=color)
            ax.legend(fontsize=6)

        for idx in range(n_cases, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Compositional Nesting ({label})", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced2_nesting_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  5. LINEAR PROBE BASELINE
# ══════════════════════════════════════════════════════════════════════

def linear_probe(model, tokenizer, sae, device, layer, site,
                 df, target_nodes, n_per_node=200, cache_dir=""):
    """Train a linear probe on activations to predict AST node type.
    Compare against SAE feature-based classification."""
    cache_path = os.path.join(cache_dir, f"linear_probe_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] linear_probe L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target_mod = get_hook_target(model, layer, site)

    # Gather labeled activations
    node_to_idx = {n: i for i, n in enumerate(target_nodes)}
    X_list, y_list = [], []
    Z_list = []  # SAE latents for feature-based classification

    for node in tqdm(target_nodes, desc=f"  Probe harvest L{layer}/{site}"):
        kw = NODE_KEYWORDS.get(node, node.lower())
        mask = df["prompt_text"].str.contains(kw, case=True, na=False, regex=False)
        matching = df[mask]["prompt_text"].tolist()
        random.seed(42)
        random.shuffle(matching)
        count = 0

        for code in matching:
            if count >= n_per_node:
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
            act = acts_store[0][0, -1, :]  # last token, [d_model]
            _, z = sae(act.unsqueeze(0))

            X_list.append(act.cpu())
            Z_list.append(z[0].cpu())
            y_list.append(node_to_idx[node])
            count += 1

    if len(X_list) < 50:
        results = {"error": "insufficient data"}
        with open(cache_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    X = torch.stack(X_list).detach().numpy()
    Z = torch.stack(Z_list).detach().numpy()
    y = np.array(y_list)

    # Train/test split (80/20)
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    Z_train, Z_test = Z[train_idx], Z[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_classes = len(target_nodes)

    # 1. Linear probe on raw activations
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(X_train, y_train)
    y_pred_probe = probe.predict(X_test)
    probe_acc = accuracy_score(y_test, y_pred_probe)
    probe_f1 = f1_score(y_test, y_pred_probe, average="weighted")

    # Per-class accuracy for probe
    probe_per_class = {}
    for node, idx in node_to_idx.items():
        node_mask = y_test == idx
        if node_mask.sum() > 0:
            node_acc = (y_pred_probe[node_mask] == idx).mean()
            probe_per_class[node] = round(float(node_acc), 4)

    # 2. Linear probe on SAE latents
    probe_z = LogisticRegression(max_iter=1000, C=1.0)
    probe_z.fit(Z_train, y_train)
    y_pred_z = probe_z.predict(Z_test)
    z_acc = accuracy_score(y_test, y_pred_z)
    z_f1 = f1_score(y_test, y_pred_z, average="weighted")

    z_per_class = {}
    for node, idx in node_to_idx.items():
        node_mask = y_test == idx
        if node_mask.sum() > 0:
            node_acc = (y_pred_z[node_mask] == idx).mean()
            z_per_class[node] = round(float(node_acc), 4)

    # 3. Naive SAE feature classifier (argmax of best feature activation)
    # Not a trained model — just checks if the target feature fires highest
    # This tests whether the "best feature" from discovery actually discriminates

    results = {
        "n_train": int(split),
        "n_test": int(n - split),
        "n_classes": n_classes,
        "raw_activation_probe": {
            "accuracy": round(float(probe_acc), 4),
            "weighted_f1": round(float(probe_f1), 4),
            "per_class_accuracy": probe_per_class,
        },
        "sae_latent_probe": {
            "accuracy": round(float(z_acc), 4),
            "weighted_f1": round(float(z_f1), 4),
            "per_class_accuracy": z_per_class,
        },
    }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_linear_probe(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])

    # Collect all probe results
    probe_data = {}
    for key in sorted(all_results):
        if not key.startswith("linear_probe_"):
            continue
        label = key.replace("linear_probe_", "")
        data = all_results[key]
        if "error" in data:
            continue
        probe_data[label] = data

    if not probe_data:
        return

    # Summary bar chart: raw vs SAE latent accuracy across all (layer, site) combos
    labels = list(probe_data.keys())
    raw_accs = [probe_data[l]["raw_activation_probe"]["accuracy"] for l in labels]
    sae_accs = [probe_data[l]["sae_latent_probe"]["accuracy"] for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall accuracy comparison
    x = np.arange(len(labels))
    w = 0.35
    axes[0].bar(x - w/2, raw_accs, w, label="Raw activations", color="#4C72B0")
    axes[0].bar(x + w/2, sae_accs, w, label="SAE latents", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Linear Probe: Raw Activations vs SAE Latents")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    for i in range(len(labels)):
        axes[0].text(i - w/2, raw_accs[i] + 0.02, f"{raw_accs[i]:.2f}",
                     ha="center", fontsize=8)
        axes[0].text(i + w/2, sae_accs[i] + 0.02, f"{sae_accs[i]:.2f}",
                     ha="center", fontsize=8)

    # Right: per-class accuracy for best combo
    best_label = max(probe_data.keys(), key=lambda l: probe_data[l]["raw_activation_probe"]["accuracy"])
    best = probe_data[best_label]
    nodes = sorted(best["raw_activation_probe"]["per_class_accuracy"].keys())
    raw_per = [best["raw_activation_probe"]["per_class_accuracy"].get(n, 0) for n in nodes]
    sae_per = [best["sae_latent_probe"]["per_class_accuracy"].get(n, 0) for n in nodes]

    x = np.arange(len(nodes))
    axes[1].bar(x - w/2, raw_per, w, label="Raw", color="#4C72B0")
    axes[1].bar(x + w/2, sae_per, w, label="SAE", color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(nodes, rotation=45, ha="right", fontsize=7)
    axes[1].set_ylabel("Per-class accuracy")
    axes[1].set_title(f"Per-class accuracy ({best_label})")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"advanced2_linear_probe.png"), dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    cfg = load_config()
    ensure_dirs(cfg)
    adv_cache = data_path("cache/advanced2")
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

    best_features = load_best_features(cfg)
    target_nodes = cfg["sae_params"]["target_nodes"]

    # Load SAEs
    print("Loading SAEs...")
    saes = {}
    for layer in range(8):
        for site in SITES:
            sae_path = get_sae_path(cfg, layer, site)
            if os.path.exists(sae_path):
                sae, meta = load_sae_from_checkpoint(sae_path, d_model, device)
                saes[(layer, site)] = sae
                print(f"  L{layer}/{site}: R²={meta.get('r2', '?')}")

    all_results = {}

    # ── Analysis 1: Causal Zero-Ablation ──
    print("\n=== Analysis 1: Causal Zero-Ablation ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"causal_ablation_{label}"] = causal_ablation(
                model, tokenizer, saes[key], device, layer, site, bf, df,
                cache_dir=adv_cache
            )

    # ── Analysis 2: Positional Analysis ──
    print("\n=== Analysis 2: Positional Analysis ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"positional_{label}"] = positional_analysis(
                model, tokenizer, saes[key], device, layer, site, bf, df,
                cache_dir=adv_cache
            )

    # ── Analysis 3: Minimal Pairs ──
    print("\n=== Analysis 3: Minimal Pairs ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"minimal_pairs_{label}"] = minimal_pairs(
                model, tokenizer, saes[key], device, layer, site, bf,
                cache_dir=adv_cache
            )

    # ── Analysis 4: Compositional Nesting ──
    print("\n=== Analysis 4: Compositional Nesting ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"nesting_{label}"] = compositional_nesting(
                model, tokenizer, saes[key], device, layer, site, bf,
                cache_dir=adv_cache
            )

    # ── Analysis 5: Linear Probe ──
    print("\n=== Analysis 5: Linear Probe ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            all_results[f"linear_probe_{label}"] = linear_probe(
                model, tokenizer, saes[key], device, layer, site, df,
                target_nodes, cache_dir=adv_cache
            )

    # ── Generate all visualizations ──
    print("\n=== Generating Visualizations ===")
    plot_causal_ablation(all_results, cfg)
    plot_positional(all_results, cfg)
    plot_minimal_pairs(all_results, cfg)
    plot_nesting(all_results, cfg)
    plot_linear_probe(all_results, cfg)

    # Save master results
    out_path = data_path("results/advanced_analysis_2.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")
    print(f"Visualizations saved to {resolve_path(cfg['paths']['dirs']['results'])}")


if __name__ == "__main__":
    main()
