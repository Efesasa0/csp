"""
Advanced SAE Analysis Suite — 6 analyses on all 16 SAEs.
1. Logit Lens per Feature
2. Feature Steering (short, 5 tokens)
3. Cross-Site Feature Alignment
4. Cross-Layer Feature Tracking
5. Co-activation Graph
6. Feature Steering for Code Generation

Loads model once. Caches everything.
"""
import os, sys, json, ast, torch, math
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import (
    load_config, get_device, get_layers, get_n_embd,
    resolve_path, data_path, ensure_dirs, patch_environment,
    get_sae_path, load_sae_from_checkpoint
)

LAYERS = [4, 5, 6, 7]
SITES = ["mlp", "resid"]

# AST node -> keyword for steering detection
NODE_KEYWORDS = {
    "For": "for ", "While": "while ", "If": "if ",
    "FunctionDef": "def ", "AsyncFunctionDef": "async def ",
    "ClassDef": "class ", "With": "with ", "Try": "try:",
    "Lambda": "lambda ", "Return": "return", "Yield": "yield",
    "ListComp": "[", "Assert": "assert ",
}


def load_best_features(cfg):
    """Load best feature per (layer, site, node) from multisite discovery results."""
    path = data_path("results/multilayer_multisite_discovery.json")
    with open(path) as f:
        data = json.load(f)

    best = {}
    for key, layer_data in data.items():
        nodes = layer_data.get("nodes", {})
        best[key] = {}
        for node, info in nodes.items():
            # Use top_by_selectivity first, fall back to winner_features
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
#  1. LOGIT LENS
# ══════════════════════════════════════════════════════════════════════

def logit_lens_per_feature(model, tokenizer, sae, layer, site, best_features, cache_dir):
    cache_path = os.path.join(cache_dir, f"logit_lens_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] logit_lens L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    unembed = model.get_output_embeddings().weight.detach().t()  # [d_model, vocab]
    results = {}

    for node, f_id in tqdm(best_features.items(), desc=f"  LogitLens L{layer}/{site}"):
        feat_dir = sae.decoder.weight[:, f_id]
        logits = torch.matmul(feat_dir, unembed)

        top_v, top_i = torch.topk(logits, k=10)
        bot_v, bot_i = torch.topk(logits, k=10, largest=False)

        results[node] = {
            "feature": f_id,
            "promoted": [
                {"token": tokenizer.decode([i.item()]).strip(), "logit": round(v.item(), 4)}
                for v, i in zip(top_v, top_i)
            ],
            "inhibited": [
                {"token": tokenizer.decode([i.item()]).strip(), "logit": round(v.item(), 4)}
                for v, i in zip(bot_v, bot_i)
            ],
        }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_logit_lens(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    for key in all_results:
        if not key.startswith("logit_lens_"):
            continue
        label = key.replace("logit_lens_", "")
        data = all_results[key]
        nodes = list(data.keys())
        if not nodes:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(nodes) * 0.5)))

        # Promoted tokens
        matrix = []
        tokens_list = []
        for node in nodes:
            promoted = data[node]["promoted"][:5]
            matrix.append([p["logit"] for p in promoted])
            if not tokens_list:
                tokens_list = [p["token"] for p in promoted]

        matrix = np.array(matrix)
        im1 = ax1.imshow(matrix, cmap="Reds", aspect="auto")
        ax1.set_yticks(range(len(nodes)))
        ax1.set_yticklabels(nodes, fontsize=8)
        ax1.set_xticks(range(len(tokens_list)))
        ax1.set_xticklabels(tokens_list, rotation=45, ha="right", fontsize=7)
        ax1.set_title(f"Promoted Tokens ({label})")
        plt.colorbar(im1, ax=ax1, shrink=0.6)

        # Add per-node token labels
        for i, node in enumerate(nodes):
            for j, p in enumerate(data[node]["promoted"][:5]):
                ax1.text(j, i, p["token"], ha="center", va="center", fontsize=6, color="black")

        # Inhibited tokens
        matrix2 = []
        tokens2 = []
        for node in nodes:
            inhibited = data[node]["inhibited"][:5]
            matrix2.append([p["logit"] for p in inhibited])
            if not tokens2:
                tokens2 = [p["token"] for p in inhibited]

        matrix2 = np.array(matrix2)
        im2 = ax2.imshow(matrix2, cmap="Blues_r", aspect="auto")
        ax2.set_yticks(range(len(nodes)))
        ax2.set_yticklabels(nodes, fontsize=8)
        ax2.set_xticks(range(len(tokens2)))
        ax2.set_xticklabels(tokens2, rotation=45, ha="right", fontsize=7)
        ax2.set_title(f"Inhibited Tokens ({label})")
        plt.colorbar(im2, ax=ax2, shrink=0.6)

        for i, node in enumerate(nodes):
            for j, p in enumerate(data[node]["inhibited"][:5]):
                ax2.text(j, i, p["token"], ha="center", va="center", fontsize=6, color="black")

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced_logit_lens_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  2. FEATURE STEERING (SHORT)
# ══════════════════════════════════════════════════════════════════════

def feature_steering(model, tokenizer, sae, device, layer, site, best_features, df,
                     multipliers=(2.0, 5.0, 10.0), n_gen=5, cache_dir=""):
    cache_path = os.path.join(cache_dir, f"steering_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] steering L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)
    results = {}

    for node, f_id in tqdm(best_features.items(), desc=f"  Steering L{layer}/{site}"):
        prompts = df[df["ast_node"] == node]["prompt_text"].tolist()[:3]
        if not prompts:
            continue

        prompt = prompts[0]
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Get baseline activation value
        acts = []
        h = target.register_forward_hook(lambda m, i, o: acts.append(
            (o[0] if isinstance(o, tuple) else o).detach()
        ))
        with torch.no_grad():
            model(input_ids=input_ids)
        h.remove()
        _, z = sae(acts[0][0, -1, :])
        original_val = z[f_id].item()

        node_results = {}
        for mult in multipliers:
            steer_val = original_val * mult

            # Generate tokens with steering
            gen_ids = input_ids.clone()
            gen_tokens = []
            for _ in range(n_gen):
                def hook_fn(m, i, o, _sv=steer_val, _fid=f_id):
                    x = o[0] if isinstance(o, tuple) else o
                    _, z = sae(x)
                    z[..., _fid] = _sv
                    new_acts = sae.decoder(z) + sae.b_dec
                    return (new_acts,) if isinstance(o, tuple) else new_acts

                hk = target.register_forward_hook(hook_fn)
                with torch.no_grad():
                    logits = model(input_ids=gen_ids).logits[0, -1]
                hk.remove()
                next_id = logits.argmax().item()
                gen_tokens.append(tokenizer.decode([next_id]))
                gen_ids = torch.cat([gen_ids, torch.tensor([[next_id]], device=device)], dim=1)

            # Baseline (no steering)
            base_ids = input_ids.clone()
            base_tokens = []
            for _ in range(n_gen):
                with torch.no_grad():
                    logits = model(input_ids=base_ids).logits[0, -1]
                next_id = logits.argmax().item()
                base_tokens.append(tokenizer.decode([next_id]))
                base_ids = torch.cat([base_ids, torch.tensor([[next_id]], device=device)], dim=1)

            steered_text = "".join(gen_tokens)
            baseline_text = "".join(base_tokens)
            kw = NODE_KEYWORDS.get(node, node.lower())
            has_keyword = kw.strip() in steered_text.lower()
            baseline_has = kw.strip() in baseline_text.lower()

            node_results[str(mult)] = {
                "steered": steered_text,
                "baseline": baseline_text,
                "has_keyword": has_keyword,
                "baseline_has_keyword": baseline_has,
                "original_val": round(original_val, 4),
                "steer_val": round(steer_val, 4),
            }

        results[node] = node_results

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


# ══════════════════════════════════════════════════════════════════════
#  3. CROSS-SITE FEATURE ALIGNMENT
# ══════════════════════════════════════════════════════════════════════

def cross_site_alignment(sae_mlp, sae_resid, bf_mlp, bf_resid, layer, cache_dir):
    cache_path = os.path.join(cache_dir, f"cross_site_L{layer}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] cross_site L{layer}")
        with open(cache_path) as f:
            return json.load(f)

    results = {}
    resid_dec = sae_resid.decoder.weight  # [d_model, n_hidden_resid]
    resid_norms = resid_dec.norm(dim=0, keepdim=True)  # [1, n_hidden_resid]

    for node in tqdm(list(bf_mlp.keys()), desc=f"  CrossSite L{layer}"):
        if node not in bf_resid:
            continue

        mlp_fid = bf_mlp[node]
        resid_fid = bf_resid[node]
        d_mlp = sae_mlp.decoder.weight[:, mlp_fid]  # [d_model]

        # Cosine sim with all resid features
        cos_sims = F.cosine_similarity(
            d_mlp.unsqueeze(1), resid_dec, dim=0
        )  # [n_hidden_resid]

        best_match_idx = cos_sims.argmax().item()
        best_match_cos = cos_sims[best_match_idx].item()
        assigned_cos = cos_sims[resid_fid].item()

        results[node] = {
            "mlp_feat": mlp_fid,
            "resid_feat": resid_fid,
            "assigned_cosine": round(assigned_cos, 4),
            "best_resid_match": best_match_idx,
            "best_resid_cosine": round(best_match_cos, 4),
        }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_cross_site(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    for key in all_results:
        if not key.startswith("cross_site_"):
            continue
        label = key.replace("cross_site_", "")
        data = all_results[key]
        nodes = list(data.keys())
        if not nodes:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(nodes))
        assigned = [data[n]["assigned_cosine"] for n in nodes]
        best = [data[n]["best_resid_cosine"] for n in nodes]

        ax.bar(x - 0.2, assigned, 0.4, label="Assigned pair", color="steelblue")
        ax.bar(x + 0.2, best, 0.4, label="Best possible match", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(nodes, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Cross-Site Alignment ({label})")
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced_cross_site_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  4. CROSS-LAYER FEATURE TRACKING
# ══════════════════════════════════════════════════════════════════════

def cross_layer_tracking(saes, best_features, site, cache_dir):
    cache_path = os.path.join(cache_dir, f"cross_layer_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] cross_layer {site}")
        with open(cache_path) as f:
            return json.load(f)

    available_layers = sorted([l for (l, s) in saes if s == site])
    nodes = set()
    for l in available_layers:
        key = f"L{l}_{site}"
        if key in best_features:
            nodes |= set(best_features[key].keys())
    nodes = sorted(nodes)

    results = {}
    for node in tqdm(nodes, desc=f"  CrossLayer {site}"):
        # Collect decoder directions across layers
        directions = {}
        for l in available_layers:
            key = f"L{l}_{site}"
            if key in best_features and node in best_features[key]:
                f_id = best_features[key][node]
                sae = saes[(l, site)]
                directions[l] = sae.decoder.weight[:, f_id].detach()

        if len(directions) < 2:
            continue

        # Build cosine similarity matrix
        layer_list = sorted(directions.keys())
        n = len(layer_list)
        sim_matrix = [[0.0] * n for _ in range(n)]
        for i, l1 in enumerate(layer_list):
            for j, l2 in enumerate(layer_list):
                sim = F.cosine_similarity(
                    directions[l1].unsqueeze(0),
                    directions[l2].unsqueeze(0)
                ).item()
                sim_matrix[i][j] = round(sim, 4)

        results[node] = {
            "layers": layer_list,
            "sim_matrix": sim_matrix,
        }

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_cross_layer(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    for key in all_results:
        if not key.startswith("cross_layer_"):
            continue
        site = key.replace("cross_layer_", "")
        data = all_results[key]
        nodes = [n for n in data if "sim_matrix" in data[n]]
        if not nodes:
            continue

        ncols = 4
        nrows = math.ceil(len(nodes) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        if nrows == 1:
            axes = [axes]
        axes_flat = [ax for row in axes for ax in (row if hasattr(row, '__len__') else [row])]

        for idx, node in enumerate(nodes):
            ax = axes_flat[idx]
            mat = np.array(data[node]["sim_matrix"])
            layers = data[node]["layers"]
            im = ax.imshow(mat, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="equal")
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
            ax.set_yticks(range(len(layers)))
            ax.set_yticklabels([f"L{l}" for l in layers], fontsize=7)
            ax.set_title(node, fontsize=9)

        for idx in range(len(nodes), len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(f"Cross-Layer Feature Tracking ({site})", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced_cross_layer_{site}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  5. CO-ACTIVATION GRAPH
# ══════════════════════════════════════════════════════════════════════

def coactivation_graph(model, tokenizer, sae, device, layer, site, best_features,
                       df, n_prompts=200, cache_dir=""):
    cache_path = os.path.join(cache_dir, f"coact_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] coactivation L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    # Collect all features of interest (union of all nodes' winners)
    all_feats = set()
    for node, f_id in best_features.items():
        all_feats.add(f_id)
    all_feats = sorted(all_feats)

    if len(all_feats) < 2:
        return {}

    # Harvest latents for random prompts
    target = get_hook_target(model, layer, site)
    prompts = df["prompt_text"].tolist()[:n_prompts]

    activations = []  # [n_prompts, n_feats] binary
    for p in tqdm(prompts, desc=f"  Coact L{layer}/{site}", leave=False):
        acts = []
        h = target.register_forward_hook(lambda m, i, o: acts.append(
            (o[0] if isinstance(o, tuple) else o).detach()
        ))
        with torch.no_grad():
            model(input_ids=tokenizer(p, return_tensors="pt").input_ids.to(device))
        h.remove()
        _, z = sae(acts[0][0, -1, :])
        binary = (z[all_feats] > 1e-5).cpu()
        activations.append(binary)

    act_matrix = torch.stack(activations).float()  # [n_prompts, n_feats]

    # Pairwise co-activation rate
    n_feats = len(all_feats)
    coact = torch.zeros(n_feats, n_feats)
    for i in range(n_feats):
        for j in range(n_feats):
            both = (act_matrix[:, i] * act_matrix[:, j]).mean().item()
            coact[i, j] = both

    # Build per-node results
    results = {}
    for node, f_id in best_features.items():
        if f_id not in all_feats:
            continue
        idx = all_feats.index(f_id)
        cofiring = []
        for j, other_feat in enumerate(all_feats):
            if j != idx:
                cofiring.append({
                    "feature": other_feat,
                    "coact_rate": round(coact[idx, j].item(), 4),
                })
        cofiring.sort(key=lambda x: -x["coact_rate"])
        results[node] = {
            "feature": f_id,
            "top_cofiring": cofiring[:10],
            "self_activation_rate": round(coact[idx, idx].item(), 4),
        }

    results["_all_features"] = all_feats
    results["_coact_matrix"] = coact.tolist()

    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_coactivation(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    for key in all_results:
        if not key.startswith("coactivation_"):
            continue
        label = key.replace("coactivation_", "")
        data = all_results[key]
        if "_coact_matrix" not in data:
            continue

        matrix = np.array(data["_coact_matrix"])
        feats = data["_all_features"]

        # Build node-name labels: map feature_id -> "Node (id)"
        feat_to_node = {}
        for node, info in data.items():
            if node.startswith("_"):
                continue
            f_id = info.get("feature")
            if f_id is not None:
                feat_to_node[f_id] = node
        tick_labels = [f"{feat_to_node.get(f, '?')} ({f})" for f in feats]

        # Zero the diagonal so off-diagonal structure is visible
        off_diag = matrix.copy()
        np.fill_diagonal(off_diag, 0)

        n = len(feats)
        fig_size = max(8, n * 0.7)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
        im = ax.imshow(off_diag, cmap="YlOrRd", aspect="equal",
                        vmin=0, vmax=max(off_diag.max(), 0.01))

        ax.set_xticks(range(n))
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n))
        ax.set_yticklabels(tick_labels, fontsize=8)

        # Annotate cells with values
        thresh = off_diag.max() * 0.6
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                val = off_diag[i, j]
                if val > 0.005:
                    color = "white" if val > thresh else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=6, color=color)

        ax.set_title(f"Feature Co-activation — off-diagonal ({label})", fontsize=12)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Co-firing rate")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced_coactivation_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  6. FEATURE STEERING FOR CODE GENERATION
# ══════════════════════════════════════════════════════════════════════

def feature_steering_codegen(model, tokenizer, sae, device, layer, site, best_features,
                             df, multipliers=(2.0, 5.0, 10.0), n_trials=20,
                             max_tokens=50, cache_dir=""):
    cache_path = os.path.join(cache_dir, f"steering_codegen_L{layer}_{site}.json")
    if os.path.exists(cache_path):
        print(f"  [cache hit] steering_codegen L{layer}/{site}")
        with open(cache_path) as f:
            return json.load(f)

    target = get_hook_target(model, layer, site)
    results = {}

    total = sum(len(df[df["ast_node"] == n]) > 0 for n in best_features) * len(multipliers) * n_trials
    pbar = tqdm(total=total, desc=f"  SteerGen L{layer}/{site}")
    for node, f_id in best_features.items():
        kw = NODE_KEYWORDS.get(node, node.lower()).strip()
        prompts = df[df["ast_node"] == node]["prompt_text"].tolist()[:n_trials]
        if not prompts:
            continue

        node_results = {}
        for mult in multipliers:
            steered_hits = 0
            baseline_hits = 0
            examples = []

            for prompt in prompts[:n_trials]:
                pbar.update(1)
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

                # Get original activation
                acts = []
                h = target.register_forward_hook(lambda m, i, o: acts.append(
                    (o[0] if isinstance(o, tuple) else o).detach()
                ))
                with torch.no_grad():
                    model(input_ids=input_ids)
                h.remove()
                _, z = sae(acts[0][0, -1, :])
                original_val = z[f_id].item()
                steer_val = original_val * mult

                # Steered generation
                gen_ids = input_ids.clone()
                gen_tokens = []
                for _ in range(max_tokens):
                    def hook_fn(m, i, o, _sv=steer_val, _fid=f_id):
                        x = o[0] if isinstance(o, tuple) else o
                        _, z = sae(x)
                        z[..., _fid] = _sv
                        new_acts = sae.decoder(z) + sae.b_dec
                        return (new_acts,) if isinstance(o, tuple) else new_acts

                    hk = target.register_forward_hook(hook_fn)
                    with torch.no_grad():
                        logits = model(input_ids=gen_ids).logits[0, -1]
                    hk.remove()
                    next_id = logits.argmax().item()
                    tok = tokenizer.decode([next_id])
                    gen_tokens.append(tok)
                    gen_ids = torch.cat([gen_ids, torch.tensor([[next_id]], device=device)], dim=1)
                    if "\n" in tok:
                        break

                # Baseline generation
                base_ids = input_ids.clone()
                base_tokens = []
                for _ in range(max_tokens):
                    with torch.no_grad():
                        logits = model(input_ids=base_ids).logits[0, -1]
                    next_id = logits.argmax().item()
                    tok = tokenizer.decode([next_id])
                    base_tokens.append(tok)
                    base_ids = torch.cat([base_ids, torch.tensor([[next_id]], device=device)], dim=1)
                    if "\n" in tok:
                        break

                steered_text = "".join(gen_tokens)
                baseline_text = "".join(base_tokens)
                if kw.lower() in steered_text.lower():
                    steered_hits += 1
                if kw.lower() in baseline_text.lower():
                    baseline_hits += 1

                if len(examples) < 3:
                    examples.append({
                        "steered": steered_text[:100],
                        "baseline": baseline_text[:100],
                    })

            node_results[str(mult)] = {
                "keyword_rate": round(steered_hits / max(len(prompts), 1), 4),
                "baseline_rate": round(baseline_hits / max(len(prompts), 1), 4),
                "n_trials": len(prompts),
                "examples": examples,
            }

        results[node] = node_results

    pbar.close()
    with open(cache_path, "w") as f:
        json.dump(results, f, indent=2)
    return results


def plot_steering_codegen(all_results, cfg):
    results_dir = resolve_path(cfg["paths"]["dirs"]["results"])
    for key in all_results:
        if not key.startswith("steering_codegen_"):
            continue
        label = key.replace("steering_codegen_", "")
        data = all_results[key]
        nodes = [n for n in data if n != "_meta"]
        if not nodes:
            continue

        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(nodes))
        width = 0.2
        mults = ["2.0", "5.0", "10.0"]
        colors = ["#4CAF50", "#FF9800", "#F44336"]

        # Baseline
        baseline_rates = [data[n].get("2.0", {}).get("baseline_rate", 0) for n in nodes]
        ax.bar(x - 1.5 * width, baseline_rates, width, label="Baseline", color="gray", alpha=0.6)

        for i, mult in enumerate(mults):
            rates = [data[n].get(mult, {}).get("keyword_rate", 0) for n in nodes]
            ax.bar(x + (i - 0.5) * width, rates, width, label=f"Steer {mult}x", color=colors[i])

        ax.set_xticks(x)
        ax.set_xticklabels(nodes, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Keyword Appearance Rate")
        ax.set_title(f"Feature Steering Code Generation ({label})")
        ax.legend()
        ax.set_ylim(0, 1.1)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"advanced_steering_codegen_{label}.png"), dpi=150)
        plt.close()


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    cfg = load_config()
    ensure_dirs(cfg)

    adv_cache = data_path("cache/advanced")
    os.makedirs(adv_cache, exist_ok=True)

    # Load model ONCE
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

    # Load best features
    best_features = load_best_features(cfg)

    # Load all SAEs
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

    # ── Phase 1: Fast analyses (tensor ops only) ──
    print("\n=== Phase 1: Logit Lens ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            print(f"  {label}...")
            all_results[f"logit_lens_{label}"] = logit_lens_per_feature(
                model, tokenizer, saes[key], layer, site, bf, adv_cache
            )

    print("\n=== Phase 2: Cross-Site Alignment ===")
    for layer in LAYERS:
        mlp_key = (layer, "mlp")
        resid_key = (layer, "resid")
        if mlp_key in saes and resid_key in saes:
            bf_mlp = best_features.get(f"L{layer}_mlp", {})
            bf_resid = best_features.get(f"L{layer}_resid", {})
            if bf_mlp and bf_resid:
                print(f"  L{layer}...")
                all_results[f"cross_site_L{layer}"] = cross_site_alignment(
                    saes[mlp_key], saes[resid_key], bf_mlp, bf_resid, layer, adv_cache
                )

    print("\n=== Phase 3: Cross-Layer Tracking ===")
    for site in SITES:
        print(f"  {site}...")
        all_results[f"cross_layer_{site}"] = cross_layer_tracking(
            saes, best_features, site, adv_cache
        )

    # ── Phase 2: Medium analyses ──
    print("\n=== Phase 4: Co-activation ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            print(f"  {label}...")
            all_results[f"coactivation_{label}"] = coactivation_graph(
                model, tokenizer, saes[key], device, layer, site, bf, df,
                cache_dir=adv_cache
            )

    # ── Phase 3: Slow analyses (generation) ──
    print("\n=== Phase 5: Feature Steering (short) ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"steering_{label}"] = feature_steering(
                model, tokenizer, saes[key], device, layer, site, bf, df,
                cache_dir=adv_cache
            )

    print("\n=== Phase 6: Feature Steering (codegen) ===")
    for layer in LAYERS:
        for site in SITES:
            key = (layer, site)
            if key not in saes:
                continue
            label = f"L{layer}_{site}"
            bf = best_features.get(label, {})
            if not bf:
                continue
            all_results[f"steering_codegen_{label}"] = feature_steering_codegen(
                model, tokenizer, saes[key], device, layer, site, bf, df,
                cache_dir=adv_cache
            )

    # ── Generate all visualizations ──
    print("\n=== Generating Visualizations ===")
    plot_logit_lens(all_results, cfg)
    plot_cross_site(all_results, cfg)
    plot_cross_layer(all_results, cfg)
    plot_coactivation(all_results, cfg)
    plot_steering_codegen(all_results, cfg)

    # ── Save master results ──
    results_path = data_path("results/advanced_analysis.json")

    # Clean non-serializable data
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

    with open(results_path, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\nAll results saved to {results_path}")
    print(f"Visualizations saved to {resolve_path(cfg['paths']['dirs']['results'])}")


if __name__ == "__main__":
    main()
