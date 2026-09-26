import torch
import torch.nn.functional as F
from tqdm import tqdm
from core import load_all, get_layers
import ast
import keyword


class DiscoveryEngine:
    def __init__(self, cfg, site="mlp", layer=None, shared=None):
        """
        Args:
            cfg: config dict
            site: "mlp" or "resid" — which activation site to analyze
            layer: override target layer (default: from config)
            shared: dict with tokenizer, model, device to avoid reloading
        """
        self.cfg = cfg
        self.site = site
        self.layer = layer if layer is not None else cfg["model"]["target_layer"]

        if shared:
            self.tokenizer = shared["tokenizer"]
            self.model = shared["model"]
            self.device = shared["device"]
            self.sae = shared.get("sae")
        else:
            self.tokenizer, self.model, self.sae, self.device = load_all(
                cfg, layer=self.layer, site=site
            )

        # If no SAE provided via shared, load for this layer+site
        if self.sae is None:
            from core import get_sae_path, load_sae_from_checkpoint, get_n_embd
            sae_path = get_sae_path(cfg, self.layer, site)
            d_model = get_n_embd(self.model.config)
            self.sae, _ = load_sae_from_checkpoint(sae_path, d_model, self.device)

        self.layers = get_layers(self.model)
        self.latent_means = None

    def _get_hook_target(self):
        """Return the module to hook based on site."""
        if self.site == "mlp":
            return self.layers[self.layer].mlp
        else:  # resid = block output
            return self.layers[self.layer]

    # ------------------------------------------------------------------ #
    #  Core                                                                #
    # ------------------------------------------------------------------ #

    def get_latents(self, prompt):
        target = self._get_hook_target()
        acts = []
        h = target.register_forward_hook(lambda m, i, o: acts.append(
            (o[0] if isinstance(o, tuple) else o).detach()
        ))
        with torch.no_grad():
            self.model(
                input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.device
                )
            )
        h.remove()
        _, latents = self.sae(acts[0][0, -1, :])
        return latents

    def compute_all_means(self, df, n_samples=200):
        all_l = []
        for p in tqdm(df["prompt_text"].tolist()[:n_samples], desc="Calculating Means"):
            all_l.append(self.get_latents(p))
        self.latent_means = torch.stack(all_l).mean(dim=0)

    # ------------------------------------------------------------------ #
    #  Feature Discovery                                                   #
    # ------------------------------------------------------------------ #

    def find_best_feature(self, df, target_node, other_nodes):
        t_prompts = df[df["ast_node"] == target_node]["prompt_text"].tolist()[:50]
        if not t_prompts:
            print(f"⚠️  Skipping {target_node}: No matching prompts found in dataset.")
            return []

        o_prompts = df[df["ast_node"].isin(other_nodes)]["prompt_text"].tolist()[:50]
        if not o_prompts:
            print(f"⚠️  Skipping {target_node}: No distractor prompts available.")
            return []

        t_freq = (
            torch.stack([self.get_latents(p) > 1e-5 for p in t_prompts]).float().mean(0)
        )
        o_freq = (
            torch.stack([self.get_latents(p) > 1e-5 for p in o_prompts]).float().mean(0)
        )

        spec = t_freq - o_freq
        threshold = self.cfg["sae_params"]["feature_selectivity_threshold"]
        winners = torch.where((spec >= threshold) & (t_freq > 0.15))[0].tolist()

        if not winners:
            print(
                f"⚠️  {target_node}: no winners at threshold {threshold}, try lowering feature_selectivity_threshold."
            )

        return sorted(winners, key=lambda x: spec[x].item(), reverse=True)

    # ------------------------------------------------------------------ #
    #  Specificity                                                         #
    # ------------------------------------------------------------------ #

    def is_specific(self, cross_row, own_node):
        """
        Returns (is_specific, own_drop, avg_other_drop, ratio).
        A feature is specific if its own-node drop is significantly larger
        than the average drop across all other nodes.
        """
        threshold = self.cfg["sae_params"].get("specificity_ratio", 1.5)
        own_drop = cross_row[own_node]["drop_zero"]
        other_drops = [v["drop_zero"] for k, v in cross_row.items() if k != own_node]
        avg_other = sum(other_drops) / len(other_drops) if other_drops else 0.0
        ratio = own_drop / (avg_other + 1e-8)
        return ratio > threshold, own_drop, avg_other, round(ratio, 4)

    # ------------------------------------------------------------------ #
    #  Ablation                                                            #
    # ------------------------------------------------------------------ #

    def _get_base_logits(self, prompt):
        with torch.no_grad():
            return self.model(
                input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.device
                )
            ).logits[0, -1]

    def _ablate_logits(self, prompt, feature_id, val):
        target = self._get_hook_target()

        def hook(m, i, o):
            x = o[0] if isinstance(o, tuple) else o
            _, z = self.sae(x)
            z[..., feature_id] = val
            new_acts = self.sae.decoder(z) + self.sae.b_dec
            return (new_acts,) if isinstance(o, tuple) else new_acts

        h = target.register_forward_hook(hook)
        with torch.no_grad():
            logits = self.model(
                input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.device
                )
            ).logits[0, -1]
        h.remove()
        return logits

    def run_ablation(self, prompt, feature_id):
        """Returns zero and mean ablation drops for argmax token."""
        base_logits = self._get_base_logits(prompt)
        tid = base_logits.argmax().item()
        base_lp = F.log_softmax(base_logits, dim=-1)[tid].item()

        zero_logits = self._ablate_logits(prompt, feature_id, 0.0)
        zero_lp = F.log_softmax(zero_logits, dim=-1)[tid].item()

        mean_val = (
            self.latent_means[feature_id].item()
            if self.latent_means is not None
            else 0.0
        )
        mean_logits = self._ablate_logits(prompt, feature_id, mean_val)
        mean_lp = F.log_softmax(mean_logits, dim=-1)[tid].item()

        return {
            "drop_zero": round(base_lp - zero_lp, 4),
            "drop_mean": round(base_lp - mean_lp, 4),
        }

    def run_patching(self, src_prompt, dest_prompt, feature_id):
        """Patches src feature value into dest prompt, returns gain on dest argmax."""
        src_val = self.get_latents(src_prompt)[feature_id].item()

        base_logits = self._get_base_logits(dest_prompt)
        tid = base_logits.argmax().item()
        base_lp = F.log_softmax(base_logits, dim=-1)[tid].item()

        patched_logits = self._ablate_logits(dest_prompt, feature_id, src_val)
        patched_lp = F.log_softmax(patched_logits, dim=-1)[tid].item()

        return round(patched_lp - base_lp, 4)

    # ------------------------------------------------------------------ #
    #  Cross-Ablation                                                      #
    # ------------------------------------------------------------------ #

    def run_cross_ablation(self, feature_id, node_prompts):
        """
        Ablates feature_id and measures zero+mean drop across all node prompts.
        Returns dict of {node_name: {drop_zero, drop_mean}}
        """
        return {
            node_name: self.run_ablation(prompt, feature_id)
            for node_name, prompt in node_prompts.items()
        }

    # ------------------------------------------------------------------ #
    #  Argmax + Validity                                                   #
    # ------------------------------------------------------------------ #

    BLOCK_KEYWORDS = {
        "for",
        "while",
        "if",
        "def",
        "class",
        "with",
        "try",
        "lambda",
        "elif",
        "else",
        "except",
        "finally",
        "async",
    }

    def is_valid_continuation(self, prompt, predicted_token):
        candidate = prompt + predicted_token
        try:
            ast.parse(candidate)
            return True, "complete"
        except SyntaxError:
            pass
        try:
            ast.parse(candidate + ":\n    pass")
            return True, "incomplete_block"
        except SyntaxError:
            pass
        try:
            ast.parse(candidate + "\n    pass")
            return True, "incomplete_expr"
        except SyntaxError:
            pass
        token = predicted_token.strip()
        if keyword.iskeyword(token):
            return True, "keyword"
        if token in {
            ":",
            "->",
            "==",
            "!=",
            "<=",
            ">=",
            "+=",
            "-=",
            "*=",
            "/=",
            "(",
            "[",
            "{",
        }:
            return True, "operator"
        return False, "invalid"

    def run_argmax_check(self, prompt, feature_id):
        base_logits = self._get_base_logits(prompt)
        base_token_id = base_logits.argmax().item()
        base_token = self.tokenizer.decode([base_token_id])

        abl_logits = self._ablate_logits(prompt, feature_id, 0.0)
        abl_token_id = abl_logits.argmax().item()
        abl_token = self.tokenizer.decode([abl_token_id])

        valid_before, reason_before = self.is_valid_continuation(prompt, base_token)
        valid_after, reason_after = self.is_valid_continuation(prompt, abl_token)

        return {
            "before": base_token.strip(),
            "reason_before": reason_before,
            "valid_before": valid_before,
            "after": abl_token.strip(),
            "reason_after": reason_after,
            "valid_after": valid_after,
            "changed": base_token_id != abl_token_id,
        }
