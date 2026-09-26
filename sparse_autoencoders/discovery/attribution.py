import torch
from core import load_all, get_layers


class PathAttributor:
    def __init__(self, cfg, shared=None):
        """
        Args:
            cfg: config dict
            shared: optional dict with pre-loaded {tokenizer, model, sae, device}
                    to avoid loading the model a second time.
        """
        self.cfg = cfg
        if shared:
            self.tokenizer = shared["tokenizer"]
            self.model = shared["model"]
            self.sae = shared["sae"]
            self.device = shared["device"]
        else:
            self.tokenizer, self.model, self.sae, self.device = load_all(cfg)
        self.layer = cfg["model"]["target_layer"]
        self.layers = get_layers(self.model)

    def get_upstream_edges(self, prompt, feature_id, k=5):
        prev_idx = self.layer - 1
        attn_mod = self.layers[prev_idx].attn

        head_outputs = []

        def hook(m, i, o):
            head_outputs.append(o.detach())

        h = attn_mod.register_forward_hook(hook)

        with torch.no_grad():
            self.model(
                input_ids=self.tokenizer(prompt, return_tensors="pt").input_ids.to(
                    self.device
                )
            )
        h.remove()

        out = head_outputs[0][0, -1, :]
        enc_dir = self.sae.encoder.weight[feature_id]
        n_heads = self.model.config.n_head
        d_head = out.shape[0] // n_heads

        scores = torch.sum(
            out.view(n_heads, d_head) * enc_dir.view(n_heads, d_head), dim=-1
        )

        top_v, top_i = torch.topk(scores, k=k)
        edges = []
        for v, i in zip(top_v, top_i):
            edges.append(
                {"source": f"L{prev_idx}H{i.item()}", "weight": float(v.item())}
            )
        return edges
