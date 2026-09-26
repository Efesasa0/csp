"""
Deep discovery: strategies to find more node-specific circuits.

1. Multi-feature circuit fingerprints (top-k features as a set)
2. Magnitude-weighted selectivity (not just binary on/off)
3. Layer 4 analysis (where features explode)
4. Pairwise contrastive discovery (node-vs-node, not node-vs-all)
5. Ensemble ablation (ablate top-k features together)
"""
import os
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from core import (
    load_config, load_all, get_layers, get_device,
    resolve_path, ensure_dirs, get_path, TopKSAE, get_n_embd
)


class DeepDiscovery:
    def __init__(self, cfg, layer=None, site="mlp", shared=None):
        """
        Args:
            cfg: config dict
            layer: target layer (default: from config)
            site: "mlp" or "resid"
            shared: dict with tokenizer, model, device to avoid reloading
        """
        self.cfg = cfg
        self.site = site
        self.layer = layer if layer is not None else cfg["model"]["target_layer"]

        if shared:
            self.tokenizer = shared["tokenizer"]
            self.model = shared["model"]
            self.device = shared["device"]
        else:
            from core import patch_environment
            patch_environment()
            self.device = get_device()
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained("openai/circuit-sparsity", trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                "openai/circuit-sparsity", trust_remote_code=True
            ).to(self.device).eval()

        # Load SAE for layer+site using checkpoint metadata
        from core import get_sae_path, load_sae_from_checkpoint, get_n_embd
        sae_path = get_sae_path(cfg, self.layer, site)
        d_model = get_n_embd(self.model.config)
        self.sae, meta = load_sae_from_checkpoint(sae_path, d_model, self.device)
        print(f"Loaded SAE for L{self.layer}/{site} (R²={meta.get('r2', '?')}, exp={meta.get('expansion', '?')}x, k={meta.get('k', '?')})")

        self.layers = get_layers(self.model)
        self.latent_cache = {}

    def _get_hook_target(self):
        """Return the module to hook based on site."""
        if self.site == "mlp":
            return self.layers[self.layer].mlp
        else:  # resid = block output
            return self.layers[self.layer]

    def get_latents(self, prompt):
        if prompt in self.latent_cache:
            return self.latent_cache[prompt]
        target = self._get_hook_target()
        acts = []
        h = target.register_forward_hook(lambda m, i, o: acts.append(
            (o[0] if isinstance(o, tuple) else o).detach()
        ))
        with torch.no_grad():
            self.model(
                input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            )
        h.remove()
        _, latents = self.sae(acts[0][0, -1, :])
        self.latent_cache[prompt] = latents
        return latents

    def _get_base_logits(self, prompt):
        with torch.no_grad():
            return self.model(
                input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            ).logits[0, -1]

    def _ablate_features(self, prompt, feature_ids, val=0.0):
        """Ablate multiple features at once."""
        target = self._get_hook_target()

        def hook(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            _, z = self.sae(x)
            for fid in feature_ids:
                z[..., fid] = val
            new_acts = self.sae.decoder(z) + self.sae.b_dec
            return (new_acts,) if isinstance(o, tuple) else new_acts

        h = target.register_forward_hook(hook)
        with torch.no_grad():
            logits = self.model(
                input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            ).logits[0, -1]
        h.remove()
        return logits

    # ------------------------------------------------------------------ #
    #  Strategy 1: Magnitude-weighted selectivity                          #
    # ------------------------------------------------------------------ #

    def magnitude_selectivity(self, df, nodes, n_prompts=50):
        """Use mean activation magnitude instead of binary firing."""
        print("\n=== Strategy 1: Magnitude-Weighted Selectivity ===")
        node_means = {}

        for node in tqdm(nodes, desc="Magnitude profiles"):
            prompts = df[df["ast_node"] == node]["prompt_text"].tolist()[:n_prompts]
            if not prompts:
                continue
            latents = torch.stack([self.get_latents(p) for p in prompts])
            node_means[node] = latents.mean(dim=0)  # mean magnitude per feature

        results = {}
        for node in nodes:
            if node not in node_means:
                continue
            others = [n for n in nodes if n != node and n in node_means]
            other_mean = torch.stack([node_means[n] for n in others]).mean(dim=0)

            # Selectivity = (own magnitude - other magnitude) / (other magnitude + eps)
            contrast = (node_means[node] - other_mean) / (other_mean.abs() + 1e-6)

            # Find features where this node's activation is significantly higher
            top_idx = torch.argsort(contrast, descending=True)[:20]
            winners = []
            for idx in top_idx:
                fid = idx.item()
                score = contrast[fid].item()
                own_mag = node_means[node][fid].item()
                if own_mag > 0.01 and score > 0.5:  # meaningful activation + contrast
                    winners.append({
                        "feature": fid,
                        "contrast_score": round(score, 4),
                        "own_magnitude": round(own_mag, 4),
                        "other_magnitude": round(other_mean[fid].item(), 4),
                    })
            results[node] = winners[:10]

        return results

    # ------------------------------------------------------------------ #
    #  Strategy 2: Multi-feature circuit fingerprints                      #
    # ------------------------------------------------------------------ #

    def circuit_fingerprints(self, df, nodes, k=5, n_prompts=50):
        """Represent each node by its top-k feature set. Measure set-level specificity."""
        print("\n=== Strategy 2: Multi-Feature Circuit Fingerprints ===")

        node_profiles = {}
        for node in tqdm(nodes, desc="Fingerprints"):
            prompts = df[df["ast_node"] == node]["prompt_text"].tolist()[:n_prompts]
            if not prompts:
                continue
            latents = torch.stack([self.get_latents(p) for p in prompts])
            mean_act = latents.mean(dim=0)
            top_k = torch.argsort(mean_act, descending=True)[:k].tolist()
            node_profiles[node] = set(top_k)

        # Measure uniqueness: how many of a node's top-k features are NOT in any other node's top-k
        results = {}
        for node, profile in node_profiles.items():
            other_features = set()
            for other, other_profile in node_profiles.items():
                if other != node:
                    other_features |= other_profile

            unique = profile - other_features
            shared = profile & other_features
            results[node] = {
                "top_k_features": sorted(profile),
                "unique_features": sorted(unique),
                "shared_features": sorted(shared),
                "uniqueness_ratio": round(len(unique) / len(profile), 4) if profile else 0,
            }

        return results

    # ------------------------------------------------------------------ #
    #  Strategy 3: Pairwise contrastive discovery                          #
    # ------------------------------------------------------------------ #

    def pairwise_contrast(self, df, nodes, n_prompts=30):
        """Find features that distinguish each pair of nodes."""
        print("\n=== Strategy 3: Pairwise Contrastive Discovery ===")
        node_latents = {}
        for node in tqdm(nodes, desc="Pairwise latents"):
            prompts = df[df["ast_node"] == node]["prompt_text"].tolist()[:n_prompts]
            if not prompts:
                continue
            node_latents[node] = torch.stack([self.get_latents(p) for p in prompts])

        # For each pair, find the feature with highest discrimination
        pair_results = {}
        active_nodes = [n for n in nodes if n in node_latents]

        for i, n1 in enumerate(active_nodes):
            for n2 in active_nodes[i+1:]:
                freq1 = (node_latents[n1] > 1e-5).float().mean(0)
                freq2 = (node_latents[n2] > 1e-5).float().mean(0)
                diff = freq1 - freq2

                # Feature most selective for n1 over n2
                best_for_n1 = torch.argmax(diff).item()
                # Feature most selective for n2 over n1
                best_for_n2 = torch.argmin(diff).item()

                pair_results[f"{n1}_vs_{n2}"] = {
                    "best_for_first": {
                        "feature": best_for_n1,
                        "freq_first": round(freq1[best_for_n1].item(), 4),
                        "freq_second": round(freq2[best_for_n1].item(), 4),
                        "gap": round(diff[best_for_n1].item(), 4),
                    },
                    "best_for_second": {
                        "feature": best_for_n2,
                        "freq_first": round(freq1[best_for_n2].item(), 4),
                        "freq_second": round(freq2[best_for_n2].item(), 4),
                        "gap": round(-diff[best_for_n2].item(), 4),
                    },
                }

        return pair_results

    # ------------------------------------------------------------------ #
    #  Strategy 4: Ensemble ablation (ablate top-k together)               #
    # ------------------------------------------------------------------ #

    def ensemble_ablation(self, df, nodes, k=5, n_prompts=30):
        """Ablate the top-k features together and measure joint causal effect."""
        print("\n=== Strategy 4: Ensemble Ablation (top-k features) ===")

        node_latents = {}
        for node in tqdm(nodes, desc="Ensemble latents"):
            prompts = df[df["ast_node"] == node]["prompt_text"].tolist()[:n_prompts]
            if not prompts:
                continue
            node_latents[node] = torch.stack([self.get_latents(p) for p in prompts])

        results = {}
        active_nodes = [n for n in nodes if n in node_latents]

        for node in tqdm(active_nodes, desc="Ensemble ablation"):
            others = [n for n in active_nodes if n != node]

            # Find top-k selective features using magnitude
            own_mean = node_latents[node].mean(0)
            other_mean = torch.stack([node_latents[n].mean(0) for n in others]).mean(0)
            contrast = own_mean - other_mean
            top_k_ids = torch.argsort(contrast, descending=True)[:k].tolist()

            # Get a prompt for this node
            prompt = df[df["ast_node"] == node].iloc[0]["prompt_text"]

            # Single-feature ablation drops
            base_logits = self._get_base_logits(prompt)
            tid = base_logits.argmax().item()
            base_lp = F.log_softmax(base_logits, dim=-1)[tid].item()

            single_drops = []
            for fid in top_k_ids:
                abl_logits = self._ablate_features(prompt, [fid])
                abl_lp = F.log_softmax(abl_logits, dim=-1)[tid].item()
                single_drops.append(round(base_lp - abl_lp, 4))

            # Ensemble ablation (all k together)
            ens_logits = self._ablate_features(prompt, top_k_ids)
            ens_lp = F.log_softmax(ens_logits, dim=-1)[tid].item()
            ensemble_drop = round(base_lp - ens_lp, 4)

            # Cross-ablation: ensemble on other nodes
            other_drops = []
            for other_node in others[:5]:
                o_prompt = df[df["ast_node"] == other_node].iloc[0]["prompt_text"]
                o_base = self._get_base_logits(o_prompt)
                o_tid = o_base.argmax().item()
                o_base_lp = F.log_softmax(o_base, dim=-1)[o_tid].item()
                o_abl = self._ablate_features(o_prompt, top_k_ids)
                o_abl_lp = F.log_softmax(o_abl, dim=-1)[o_tid].item()
                other_drops.append(o_base_lp - o_abl_lp)

            avg_other = sum(other_drops) / len(other_drops) if other_drops else 0
            ratio = ensemble_drop / (avg_other + 1e-8)

            # Argmax change check
            ens_token = self.tokenizer.decode([ens_logits.argmax().item()]).strip()
            base_token = self.tokenizer.decode([tid]).strip()

            results[node] = {
                "top_k_features": top_k_ids,
                "single_drops": dict(zip(top_k_ids, single_drops)),
                "ensemble_drop": ensemble_drop,
                "sum_single_drops": round(sum(single_drops), 4),
                "superadditivity": round(ensemble_drop - sum(single_drops), 4),
                "avg_other_drop": round(avg_other, 4),
                "specificity_ratio": round(ratio, 4),
                "is_specific": ratio > 1.5,
                "argmax_before": base_token,
                "argmax_after": ens_token,
                "argmax_changed": base_token != ens_token,
            }

        return results


def run_deep_discovery():
    cfg = load_config()
    ensure_dirs(cfg)
    df = pd.read_parquet(resolve_path(cfg["paths"]["dataset"]))
    nodes = cfg["sae_params"]["target_nodes"]
    out_dir = resolve_path(cfg["paths"]["dirs"]["reports"])

    # Load model once, share across all DeepDiscovery instances
    from core import patch_environment
    patch_environment()
    device = get_device()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("openai/circuit-sparsity", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    ).to(device).eval()
    shared = {"tokenizer": tokenizer, "model": model, "device": device}

    all_results = {}

    # Run all combinations of (layer, site)
    combos = [
        (7, "mlp"), (7, "resid"),
        (4, "mlp"), (4, "resid"),
    ]

    for layer, site in combos:
        label = f"L{layer}_{site}"
        print("\n" + "=" * 60)
        print(f"DEEP DISCOVERY — {label.upper()}")
        print("=" * 60)

        dd = DeepDiscovery(cfg, layer=layer, site=site, shared=shared)

        mag = dd.magnitude_selectivity(df, nodes)
        all_results[f"magnitude_{label}"] = mag

        fp = dd.circuit_fingerprints(df, nodes, k=5)
        all_results[f"fingerprints_{label}"] = fp

        if site == "mlp" and layer == 7:
            pw = dd.pairwise_contrast(df, nodes)
            all_results[f"pairwise_{label}"] = pw

        ens = dd.ensemble_ablation(df, nodes, k=5)
        all_results[f"ensemble_{label}"] = ens

    # Save
    out_path = os.path.join(out_dir, "deep_discovery_report.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ------------------------------------------------------------------ #
    #  Print summary                                                       #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 70)
    print("DEEP DISCOVERY SUMMARY")
    print("=" * 70)

    for layer, site in combos:
        label = f"L{layer}_{site}"
        ens_key = f"ensemble_{label}"
        if ens_key not in all_results:
            continue

        print(f"\n--- Ensemble Ablation ({label}, k=5): Joint Specificity ---")
        for node, info in all_results[ens_key].items():
            sp = "SPECIFIC" if info["is_specific"] else "shared"
            chg = "CHANGED" if info["argmax_changed"] else "same"
            print(f"  {node:<18} drop={info['ensemble_drop']:<8} "
                  f"ratio={info['specificity_ratio']:<8} [{sp}] "
                  f"argmax [{chg}]")

    # Count specific nodes per combo
    print(f"\n--- Specific Node Counts ---")
    for layer, site in combos:
        label = f"L{layer}_{site}"
        ens_key = f"ensemble_{label}"
        if ens_key not in all_results:
            continue
        n_specific = sum(1 for v in all_results[ens_key].values() if v.get("is_specific"))
        print(f"  {label:<12} {n_specific} / {len(all_results[ens_key])} nodes specific")

    print(f"\nFull results: {out_path}")


if __name__ == "__main__":
    run_deep_discovery()
