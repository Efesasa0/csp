"""Model and SAE loading utilities."""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import get_device, patch_environment, resolve_path
from .model import TopKSAE


def get_layers(model):
    if hasattr(model, "transformer"):
        return model.transformer.h
    if hasattr(model, "h"):
        return model.h
    if hasattr(model, "circuit_model"):
        inner = model.circuit_model
        if hasattr(inner, "transformer"):
            return inner.transformer.h
        if hasattr(inner, "h"):
            return inner.h
    raise AttributeError(
        f"Structure mismatch. Top-level modules: {list(model._modules.keys())}"
    )


def get_n_embd(config):
    for attr in ["n_embd", "hidden_size", "d_model", "n_hidden", "hidden_dim"]:
        if hasattr(config, attr):
            return getattr(config, attr)
    raise AttributeError(
        f"Cannot determine hidden dimension. Config attrs: "
        f"{[a for a in dir(config) if not a.startswith('_')]}"
    )


def get_sae_path(cfg, layer, site="mlp"):
    """Get the SAE checkpoint path for a given layer and site."""
    sae_dir = cfg["paths"]["sae_dir"] if cfg else "data/saes"
    if site == "mlp":
        filename = f"sae_layer_{layer}.pt"
    else:
        filename = f"sae_layer_{layer}_{site}.pt"
    return resolve_path(os.path.join(sae_dir, filename))


def load_sae_from_checkpoint(sae_path, d_model, device,
                              fallback_expansion=8, fallback_k=32):
    """Load an SAE from a checkpoint, auto-detecting expansion/k from __meta__."""
    state = torch.load(sae_path, map_location=device, weights_only=False)
    meta = state.pop("__meta__", {})
    expansion = meta.get("expansion", fallback_expansion)
    k = meta.get("k", fallback_k)
    sae = TopKSAE(n_input=d_model, n_hidden=d_model * expansion, k=k).to(device)
    sae.load_state_dict(state)
    sae.eval()
    return sae, meta


def load_all(cfg=None, layer=7, load_weights=True, site="mlp"):
    """Load model, tokenizer, and SAE in one call."""
    device = get_device()
    patch_environment()

    if cfg is not None:
        layer = cfg["model"]["target_layer"]

    tokenizer = AutoTokenizer.from_pretrained(
        "openai/circuit-sparsity", trust_remote_code=True
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            "openai/circuit-sparsity", trust_remote_code=True
        )
        .to(device)
        .eval()
    )

    n_input = get_n_embd(model.config)
    sae_path = get_sae_path(cfg, layer, site)

    if load_weights and os.path.exists(sae_path):
        try:
            sae, meta = load_sae_from_checkpoint(sae_path, n_input, device)
            print(f"Loaded SAE from {sae_path} "
                  f"(R²={meta.get('r2', '?')}, exp={meta.get('expansion', '?')}x, "
                  f"k={meta.get('k', '?')})")
            return tokenizer, model, sae, device
        except RuntimeError:
            print(f"Architecture mismatch in {sae_path}. Starting with fresh weights.")

    expansion = cfg["sae_params"]["expansion"] if cfg else 8
    k = cfg["sae_params"]["k"] if cfg else 32
    sae = TopKSAE(n_input=n_input, n_hidden=n_input * expansion, k=k).to(device)
    return tokenizer, model, sae, device
