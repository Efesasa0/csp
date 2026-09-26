"""
01_extraction.py

Step 1 of the AST x builtin mechanistic interpretability pipeline.

Loads the circuit_sparsity model, runs every prompt in the input dataset,
and extracts five families of activations / structure:

  A. Residual stream — final layer   (n_prompts, hidden_dim)
  B. Residual stream — all layers    (n_prompts, n_layers+1, hidden_dim)
  C. Logit attribution — per head    (n_prompts, n_layers, n_heads)  +  MLP  (n_prompts, n_layers)
  D. MLP pre-activation neurons      (n_prompts, n_layers, 4*hidden_dim)   [AbsTopK input]
  E. Edge connectivity               (static, one-time)  — per-weight sparsity graph

Outputs (all written to cfg.dir):
  01_<stem>_residual_final.npy       float32  (N, H)
  01_<stem>_residual_all.npy         float32  (N, L+1, H)
  01_<stem>_head_attr.npy            float32  (N, L, n_heads)
  01_<stem>_mlp_attr.npy             float32  (N, L)
  01_<stem>_mlp_neurons.npy          float32  (N, L, 4H)   [sparse — mostly zeros]
  01_<stem>_meta.json                prompt metadata (prompt_id, ast_node, builtin_obj, …)
  01_<stem>_edge_graph.npz           static weight-sparsity adjacency (see below)

Edge graph format (npz):
  layers      int32   (E,)   source layer index  (-1 = embedding)
  src_units   int32   (E,)   source unit index within layer
  dst_layers  int32   (E,)   destination layer index
  dst_units   int32   (E,)   destination unit index
  weights     float32 (E,)   weight value (non-zero only)
  node_types  str     per-unit type tag (embedding / attn_head / mlp_neuron / lm_head)

Notes
-----
- Runs on CPU by default (circuit_sparsity is small); pass device cuda:0 if available.
- Already-extracted prompts are skipped if partial outputs exist (resume-safe).
- The edge graph is model-static: extracted once and reused across runs.
"""

from __future__ import annotations

import gc
import inspect
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from ._config import StepConfig


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID = "openai/circuit-sparsity"


def load_model(device: str = "cpu"):
    """
    Load circuit_sparsity model and tokenizer with the GPTConfig patch
    used in all QQQ/XXX notebooks to silence unknown-kwarg warnings.
    """
    subprocess.run(
        ["pip", "install", "-q", "git+https://github.com/openai/circuit_sparsity"],
        capture_output=True,
    )

    for key in list(sys.modules.keys()):
        if "circuit_sparsity" in key or "transformers_modules" in key:
            del sys.modules[key]
    importlib.invalidate_caches()

    import circuit_sparsity.inference.gpt as _gpt_mod
    _orig_init  = _gpt_mod.GPTConfig.__init__
    _known      = set(inspect.signature(_orig_init).parameters.keys()) - {"self"}

    def _patched(self, *a, **kw):
        _orig_init(self, *a, **{k: v for k, v in kw.items() if k in _known})

    if not getattr(_gpt_mod.GPTConfig.__init__, "__patched__", False):
        _patched.__patched__ = True
        _gpt_mod.GPTConfig.__init__ = _patched
        sys.modules["circuit_sparsity.gpt"] = _gpt_mod

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map=device if device != "cpu" else "cpu",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    print(f"Model loaded on {device}  [{type(model).__name__}]")
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Model geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def model_geometry(model) -> dict:
    cfg = model.circuit_model.config
    t   = model.circuit_model.transformer
    h   = t.h[0]
    mlp_dim = h.mlp.c_fc.weight.shape[0]   # 4 * hidden_dim typically
    return {
        "n_layers":   len(t.h),
        "n_heads":    cfg.n_head,
        "hidden_dim": t.wte.weight.shape[1],
        "head_dim":   t.wte.weight.shape[1] // cfg.n_head,
        "mlp_dim":    mlp_dim,
        "vocab_size": t.wte.weight.shape[0],
        "has_bigram": getattr(cfg, "enable_bigram_table", False),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-prompt extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_prompt(text: str, model, tokenizer, geo: dict, max_tokens: int = 128):
    """
    Single forward pass — returns all five extraction targets for one prompt.

    Returns
    -------
    residual_all  : (n_layers+1, hidden_dim)  float32
    head_attr     : (n_layers, n_heads)        float32  signed logit attribution
    mlp_attr      : (n_layers,)                float32  signed logit attribution
    mlp_neurons   : (n_layers, mlp_dim)        float32  pre-AbsTopK activations
    """
    import torch

    inputs    = tokenizer(text, return_tensors="pt",
                          truncation=True, max_length=max_tokens)
    input_ids = inputs["input_ids"].to(model.device)

    n_layers  = geo["n_layers"]
    n_heads   = geo["n_heads"]
    hidden    = geo["hidden_dim"]
    head_dim  = geo["head_dim"]
    mlp_dim   = geo["mlp_dim"]
    t         = model.circuit_model.transformer

    # ── storage ───────────────────────────────────────────────────────────────
    residuals   = []   # filled by hooks: n_layers+1 vectors of shape (hidden,)
    attn_inputs = []   # pre-projection inputs to c_proj:  (n_heads*head_dim,)
    mlp_outputs = []   # c_proj outputs (= MLP contribution to residual): (hidden,)
    mlp_pre     = []   # c_fc outputs (pre-AbsTopK): (mlp_dim,)
    hooks       = []

    # A/B — residual stream (before each block + before ln_f)
    for block in t.h:
        def _res_pre(m, args, buf=residuals):
            buf.append(args[0][0, -1, :].detach().cpu())
        hooks.append(block.register_forward_pre_hook(_res_pre))

    def _lnf_pre(m, args, buf=residuals):
        buf.append(args[0][0, -1, :].detach().cpu())
    hooks.append(t.ln_f.register_forward_pre_hook(_lnf_pre))

    # C — logit attribution: capture pre-c_proj attn input & mlp output
    for block in t.h:
        def _attn_in(m, inp, out, buf=attn_inputs):
            buf.append(inp[0][0, -1, :].detach().cpu())     # (n_heads*head_dim,)
        def _mlp_out(m, inp, out, buf=mlp_outputs):
            buf.append(out[0, -1, :].detach().cpu())        # (hidden,)
        hooks.append(block.attn.c_proj.register_forward_hook(_attn_in))
        hooks.append(block.mlp.c_proj.register_forward_hook(_mlp_out))

    # D — MLP pre-activation (AbsTopK input = c_fc output)
    for block in t.h:
        def _mlp_fc(m, inp, out, buf=mlp_pre):
            buf.append(out[0, -1, :].detach().cpu())        # (mlp_dim,)
        hooks.append(block.mlp.c_fc.register_forward_hook(_mlp_fc))

    # forward
    with torch.no_grad():
        out = model(input_ids=input_ids)
    for h in hooks:
        h.remove()

    # ── logit attribution (C) ──────────────────────────────────────────────────
    # W_U: (vocab, hidden);  target = argmax of last-token logits
    W_U       = model.circuit_model.lm_head.weight.detach().cpu()
    target_id = out.logits[0, -1, :].argmax().item()
    w_u       = W_U[target_id]                               # (hidden,)

    # Cache W_O per layer: (hidden, n_heads, head_dim)
    W_O_list = [
        block.attn.c_proj.weight.detach().cpu()
            .reshape(hidden, n_heads, head_dim)
        for block in t.h
    ]

    head_attr_rows = []
    for layer_idx, ai in enumerate(attn_inputs):
        x2d      = ai.view(n_heads, head_dim)                     # (n_heads, head_dim)
        # contribution of head h = W_O[:, h, :] @ x2d[h]  →  (hidden,)
        # dot with w_u = scalar attribution per head
        contribs = torch.einsum("ihd,hd->ih", W_O_list[layer_idx], x2d)  # (hidden, n_heads)
        head_attr_rows.append((w_u @ contribs).numpy())          # (n_heads,)

    mlp_attr_vals = [(w_u @ m).item() for m in mlp_outputs]

    # ── assemble ──────────────────────────────────────────────────────────────
    residual_all = np.stack([v.numpy() for v in residuals]).astype(np.float32)
    head_attr    = np.array(head_attr_rows, dtype=np.float32)      # (L, n_heads)
    mlp_attr     = np.array(mlp_attr_vals,  dtype=np.float32)      # (L,)
    mlp_neurons  = np.stack([v.numpy() for v in mlp_pre]).astype(np.float32)  # (L, mlp_dim)

    return residual_all, head_attr, mlp_attr, mlp_neurons


# ─────────────────────────────────────────────────────────────────────────────
# Edge connectivity (static — run once per model)
# ─────────────────────────────────────────────────────────────────────────────

def extract_edge_graph(model, geo: dict, eps: float = 1e-6) -> dict:
    """
    Build the weight-sparsity graph from the model's named parameters.

    Only non-zero weights (abs > eps) are included as edges.
    Node taxonomy:
      - embed          : token/position embedding rows
      - attn_q/k/v     : query/key/value projection input units
      - attn_out       : attention output projection units (= residual writers)
      - mlp_fc         : MLP c_fc units  (pre-AbsTopK)
      - mlp_proj       : MLP c_proj units (post-AbsTopK, residual writers)
      - lm_head        : unembedding rows

    Returns a dict with numpy arrays ready for np.savez().
    """
    import torch

    edges = {
        "src_name":  [],   # human-readable node name
        "dst_name":  [],
        "src_idx":   [],   # unit index within its layer/module
        "dst_idx":   [],
        "weight":    [],
        "layer":     [],   # transformer layer index (-1 = embedding / lm_head)
        "param_key": [],   # e.g. "transformer.h.2.attn.c_proj.weight"
    }

    t = model.circuit_model.transformer

    def _add_edges(param_name: str, W: np.ndarray, layer_idx: int,
                   src_tag: str, dst_tag: str):
        """Register all non-zero (i,j) entries of weight matrix W."""
        nz_dst, nz_src = np.nonzero(np.abs(W) > eps)
        for d, s, v in zip(nz_dst, nz_src, W[nz_dst, nz_src]):
            edges["src_name"].append(f"L{layer_idx}_{src_tag}_{s}")
            edges["dst_name"].append(f"L{layer_idx}_{dst_tag}_{d}")
            edges["src_idx"].append(int(s))
            edges["dst_idx"].append(int(d))
            edges["weight"].append(float(v))
            edges["layer"].append(layer_idx)
            edges["param_key"].append(param_name)

    print("  Extracting edge graph from weight matrices ...")

    # Token embedding  (vocab, hidden)
    W_emb = t.wte.weight.detach().cpu().numpy()
    nz    = np.nonzero(np.abs(W_emb) > eps)
    for tok, dim in zip(nz[0], nz[1]):
        edges["src_name"].append(f"tok_{tok}")
        edges["dst_name"].append(f"embed_dim_{dim}")
        edges["src_idx"].append(int(tok))
        edges["dst_idx"].append(int(dim))
        edges["weight"].append(float(W_emb[tok, dim]))
        edges["layer"].append(-1)
        edges["param_key"].append("transformer.wte.weight")

    for li, block in enumerate(t.h):
        # Attention QKV  (3*hidden, hidden)  →  split into Q, K, V
        if hasattr(block.attn, "c_attn"):
            W_qkv = block.attn.c_attn.weight.detach().cpu().numpy()
            h = geo["hidden_dim"]
            for tag, chunk in zip(["attn_q", "attn_k", "attn_v"],
                                   [W_qkv[:h], W_qkv[h:2*h], W_qkv[2*h:]]):
                _add_edges(f"h.{li}.attn.c_attn.weight", chunk, li, "embed", tag)

        # Attention output projection  (hidden, hidden)
        W_ao = block.attn.c_proj.weight.detach().cpu().numpy()
        _add_edges(f"h.{li}.attn.c_proj.weight", W_ao, li, "attn_out", "residual")

        # MLP c_fc  (mlp_dim, hidden)
        W_fc = block.mlp.c_fc.weight.detach().cpu().numpy()
        _add_edges(f"h.{li}.mlp.c_fc.weight", W_fc, li, "residual", "mlp_fc")

        # MLP c_proj  (hidden, mlp_dim)
        W_mp = block.mlp.c_proj.weight.detach().cpu().numpy()
        _add_edges(f"h.{li}.mlp.c_proj.weight", W_mp, li, "mlp_fc", "mlp_proj")

    # LM head  (vocab, hidden)
    W_lm = model.circuit_model.lm_head.weight.detach().cpu().numpy()
    nz   = np.nonzero(np.abs(W_lm) > eps)
    for tok, dim in zip(nz[0], nz[1]):
        edges["src_name"].append(f"residual_dim_{dim}")
        edges["dst_name"].append(f"lm_head_{tok}")
        edges["src_idx"].append(int(dim))
        edges["dst_idx"].append(int(tok))
        edges["weight"].append(float(W_lm[tok, dim]))
        edges["layer"].append(geo["n_layers"])
        edges["param_key"].append("lm_head.weight")

    total_edges = len(edges["weight"])
    total_params = sum(p.numel() for p in model.circuit_model.parameters())
    sparsity = 1.0 - total_edges / total_params
    print(f"  Edge graph: {total_edges:,} non-zero edges "
          f"/ {total_params:,} params  (sparsity {sparsity:.1%})")

    return {k: np.array(v) for k, v in edges.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_prompts(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    else:
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(cfg: StepConfig) -> None:
    in_path = cfg.input_path
    assert in_path.exists(), f"Input not found: {in_path}"

    stem   = in_path.stem                   # e.g. "contrastive_stubs"
    outdir = cfg.dir
    prefix = f"01_{stem}_"

    # ── output paths ──────────────────────────────────────────────────────────
    p_resid_final = outdir / f"{prefix}residual_final.npy"
    p_resid_all   = outdir / f"{prefix}residual_all.npy"
    p_head_attr   = outdir / f"{prefix}head_attr.npy"
    p_mlp_attr    = outdir / f"{prefix}mlp_attr.npy"
    p_mlp_neurons = outdir / f"{prefix}mlp_neurons.npy"
    p_meta        = outdir / f"{prefix}meta.json"
    p_edge        = outdir / f"{prefix}edge_graph.npz"

    # ── load prompts ──────────────────────────────────────────────────────────
    print(f"Loading prompts from {in_path} ...")
    prompts = load_prompts(in_path)
    N = len(prompts)
    print(f"  {N} prompts")

    # ── resume: check how many already extracted ───────────────────────────────
    start_idx = 0
    existing_resid = []
    existing_resid_all = []
    existing_head = []
    existing_mlp_attr = []
    existing_mlp_neurons = []

    if p_resid_final.exists():
        existing_resid = list(np.load(p_resid_final))
        start_idx = len(existing_resid)
        existing_resid_all   = list(np.load(p_resid_all))
        existing_head        = list(np.load(p_head_attr))
        existing_mlp_attr    = list(np.load(p_mlp_attr))
        existing_mlp_neurons = list(np.load(p_mlp_neurons))
        print(f"  Resuming from prompt {start_idx}/{N}")

    if start_idx >= N:
        print("  All prompts already extracted.")
    else:
        # ── load model ────────────────────────────────────────────────────────
        model, tokenizer = load_model(cfg.device)
        geo = model_geometry(model)
        print(f"  Geometry: {geo}")

        resid_final_buf   = existing_resid
        resid_all_buf     = existing_resid_all
        head_attr_buf     = existing_head
        mlp_attr_buf      = existing_mlp_attr
        mlp_neurons_buf   = existing_mlp_neurons

        for i, record in enumerate(prompts[start_idx:], start=start_idx):
            text = record.get("prompt_text") or record.get("stub", "")
            if not text:
                continue

            r_all, h_attr, m_attr, m_neu = extract_prompt(
                text, model, tokenizer, geo, cfg.max_tokens
            )

            resid_final_buf.append(r_all[-1])   # last layer only
            resid_all_buf.append(r_all)
            head_attr_buf.append(h_attr)
            mlp_attr_buf.append(m_attr)
            mlp_neurons_buf.append(m_neu)

            if (i + 1) % cfg.batch_size == 0 or i == N - 1:
                # ── checkpoint ────────────────────────────────────────────────
                np.save(p_resid_final,   np.array(resid_final_buf,   dtype=np.float32))
                np.save(p_resid_all,     np.array(resid_all_buf,     dtype=np.float32))
                np.save(p_head_attr,     np.array(head_attr_buf,     dtype=np.float32))
                np.save(p_mlp_attr,      np.array(mlp_attr_buf,      dtype=np.float32))
                np.save(p_mlp_neurons,   np.array(mlp_neurons_buf,   dtype=np.float32))
                print(f"  [{i+1}/{N}] checkpointed", end="\r")

            if (i + 1) % 50 == 0:
                gc.collect()

        print(f"\n  Extraction complete: {len(resid_final_buf)} prompts")

        # ── edge graph (static, once) ──────────────────────────────────────────
        if not cfg.skip_edge_graph:
            if p_edge.exists():
                print(f"  Edge graph already exists: {p_edge}")
            else:
                edge_data = extract_edge_graph(model, geo)
                np.savez(p_edge, **edge_data)
                print(f"  Edge graph saved -> {p_edge}")

    # ── write metadata ────────────────────────────────────────────────────────
    meta_fields = ["prompt_id", "ast_node", "builtin_obj", "variation_id",
                   "variant_type", "category", "ast_group", "builtin_group",
                   "token_length", "ast_verified", "sequence_loss", "note"]
    meta = [
        {k: r.get(k) for k in meta_fields}
        for r in prompts[:len(existing_resid) if start_idx >= N
                         else len(resid_final_buf)]  # type: ignore[possibly-undefined]
    ]
    with open(p_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print(f"  Metadata saved -> {p_meta}")

    # ── final summary ─────────────────────────────────────────────────────────
    print()
    print("Output files:")
    for p in [p_resid_final, p_resid_all, p_head_attr,
              p_mlp_attr, p_mlp_neurons, p_meta, p_edge]:
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.name:<45s}  {size_mb:6.1f} MB")

    print()
    if p_resid_all.exists():
        arr = np.load(p_resid_all)
        n, n_layers_p1, h = arr.shape
        print(f"Array shapes:")
        print(f"  residual_final : ({n}, {h})")
        print(f"  residual_all   : ({n}, {n_layers_p1}, {h})")
        if p_head_attr.exists():
            ha = np.load(p_head_attr)
            print(f"  head_attr      : {ha.shape}   (n_prompts, n_layers, n_heads)")
        if p_mlp_attr.exists():
            ma = np.load(p_mlp_attr)
            print(f"  mlp_attr       : {ma.shape}   (n_prompts, n_layers)")
        if p_mlp_neurons.exists():
            mn = np.load(p_mlp_neurons)
            print(f"  mlp_neurons    : {mn.shape}   (n_prompts, n_layers, mlp_dim)")
        if p_edge.exists():
            eg = np.load(p_edge)
            print(f"  edge_graph     : {len(eg['weight']):,} edges")

    print()
    print("Load in downstream scripts:")
    print(f"  resid_final  = np.load('{p_resid_final.name}')  # (N, H)")
    print(f"  resid_all    = np.load('{p_resid_all.name}')    # (N, L+1, H)")
    print(f"  head_attr    = np.load('{p_head_attr.name}')    # (N, L, n_heads)")
    print(f"  mlp_attr     = np.load('{p_mlp_attr.name}')     # (N, L)")
    print(f"  mlp_neurons  = np.load('{p_mlp_neurons.name}')  # (N, L, mlp_dim)")
    print(f"  meta         = json.load(open('{p_meta.name}'))")
    print(f"  edge_graph   = np.load('{p_edge.name}')         # dict of edge arrays")
