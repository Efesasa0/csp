"""
Extraction Pipeline
Visual walkthrough of 01_extraction.py — the five families of activations
it extracts, how hooks work, and what artifact files are produced.
"""
import streamlit as st

st.set_page_config(page_title="Extraction Pipeline", layout="wide")

st.title("Extraction Pipeline")

st.markdown(
    """
    **`01_extraction.py`** runs every prompt through `circuit-sparsity` and
    captures five families of activations using PyTorch forward hooks.
    The outputs feed every downstream analysis step (02 -- 06).
    """
)

# ── Key metrics ──────────────────────────────────────────────────────────────

c1, c2, c3 = st.columns(3)
c1.metric("Extraction Families", 5)
c2.metric("Prompts (contrastive set)", "~8 900")
c3.metric("Output Files", 7)

st.divider()

# ── Extraction flow diagram ──────────────────────────────────────────────────

st.header("Extraction Flow")

st.graphviz_chart(
    """
    digraph extraction {
        rankdir=BT;
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", fontname="Helvetica",
              fontsize=11, margin="0.15,0.08"];
        edge [color="#666666", arrowsize=0.7];

        /* ── inputs ──────────────────────────── */
        prompts [label="Input Prompts\\n(JSONL / Parquet)", fillcolor="#e8e8e8",
                 shape=plaintext, fontsize=12, fontcolor="#333333"];
        tok     [label="Tokenizer\\nAutoTokenizer", fillcolor="#d4e6f1"];
        prompts -> tok;

        /* ── forward pass ────────────────────── */
        subgraph cluster_fwd {
            label="Forward Pass with Hooks";
            labeljust=l;
            fontname="Helvetica"; fontsize=12;
            style="dashed"; color="#888888";
            margin=16;

            res_stream [label="Residual Stream\\n(pre-block hooks)",
                        fillcolor="#eafaf1", fontcolor="#1e8449"];
            attn_proj  [label="Attention c_proj\\n(forward hook)",
                        fillcolor="#d5f5e3"];
            mlp_proj   [label="MLP c_proj\\n(forward hook)",
                        fillcolor="#d5f5e3"];
            mlp_fc     [label="MLP c_fc\\n(forward hook)",
                        fillcolor="#d5f5e3"];
            ln_f       [label="ln_f\\n(pre-hook)",
                        fillcolor="#fdebd0", fontcolor="#b9770e"];

            res_stream -> ln_f [style=invis, weight=10];
        }

        tok -> res_stream;
        tok -> attn_proj  [style=invis, weight=0];
        tok -> mlp_proj   [style=invis, weight=0];
        tok -> mlp_fc     [style=invis, weight=0];

        /* ── five output branches ────────────── */
        fam_ab [label="A/B  Residual Stream\\nresidual_final.npy\\nresidual_all.npy",
                fillcolor="#d4e6f1"];
        fam_c1 [label="C  Head Attribution\\nhead_attr.npy",
                fillcolor="#d4e6f1"];
        fam_c2 [label="C  MLP Attribution\\nmlp_attr.npy",
                fillcolor="#d4e6f1"];
        fam_d  [label="D  MLP Neurons\\nmlp_neurons.npy",
                fillcolor="#d4e6f1"];
        fam_e  [label="E  Edge Graph\\nedge_graph.npz",
                fillcolor="#fadbd8", fontcolor="#922b21"];

        res_stream -> fam_ab;
        ln_f       -> fam_ab;
        attn_proj  -> fam_c1;
        mlp_proj   -> fam_c2;
        mlp_fc     -> fam_d;

        /* edge graph comes from params directly */
        params [label="Model Parameters\\n(direct inspection)",
                fillcolor="#fdebd0", fontcolor="#b9770e"];
        params -> fam_e [style=dashed, label="  static\\n  (no hook)",
                         fontsize=9, fontname="Helvetica"];
    }
    """,
    use_container_width=True,
)

st.divider()

# ── The Five Families ────────────────────────────────────────────────────────

st.header("The Five Extraction Families")

# --- A/B: Residual Stream ---
with st.container(border=True):
    st.subheader("A/B -- Residual Stream")
    st.markdown(
        """
        **What it captures:** The hidden state at the last token position
        *before* each transformer block (layers 0..L-1) and before `ln_f`
        (layer L). Family A is just the final layer; Family B is all layers.

        | | Shape | Notes |
        |---|---|---|
        | `residual_final` | `(N, H)` | Last-layer residual only |
        | `residual_all` | `(N, L+1, H)` | Full layer stack |
        """
    )
    st.markdown("**Hook target:** `block.register_forward_pre_hook` + `ln_f.register_forward_pre_hook`")
    st.code(
        "# A/B — residual stream hooks\n"
        "for block in transformer.h:\n"
        "    def _res_pre(m, args, buf=residuals):\n"
        "        buf.append(args[0][0, -1, :].detach().cpu())\n"
        "    block.register_forward_pre_hook(_res_pre)\n\n"
        "# final: before ln_f\n"
        "def _lnf_pre(m, args, buf=residuals):\n"
        "    buf.append(args[0][0, -1, :].detach().cpu())\n"
        "transformer.ln_f.register_forward_pre_hook(_lnf_pre)",
        language="python",
    )

# --- C: Head Attribution ---
with st.container(border=True):
    st.subheader("C -- Head Attribution (logit contribution per head)")
    st.markdown(
        """
        **What it captures:** Signed logit contribution of each attention head
        to the predicted token. Computed by capturing the *input* to `c_proj`,
        then projecting through W_O and dotting with the unembedding row for
        the argmax token.

        | | Shape |
        |---|---|
        | `head_attr` | `(N, L, n_heads)` |
        """
    )
    st.markdown("**Hook target:** `attn.c_proj.register_forward_hook` (captures input)")
    st.code(
        "# C — head attribution hook\n"
        "def _attn_in(m, inp, out, buf=attn_inputs):\n"
        "    buf.append(inp[0][0, -1, :].detach().cpu())  # (n_heads*head_dim,)\n"
        "block.attn.c_proj.register_forward_hook(_attn_in)\n\n"
        "# Post-hoc: project through W_O and dot with W_U\n"
        "x2d = attn_input.view(n_heads, head_dim)\n"
        "contribs = einsum('ihd,hd->ih', W_O, x2d)   # (hidden, n_heads)\n"
        "head_attr = w_u @ contribs                    # (n_heads,)",
        language="python",
    )

# --- C: MLP Attribution ---
with st.container(border=True):
    st.subheader("C -- MLP Attribution (logit contribution per MLP block)")
    st.markdown(
        """
        **What it captures:** Signed logit contribution of each MLP sub-layer
        to the predicted token. The `c_proj` output (MLP's residual-stream
        contribution) is dotted with the unembedding row.

        | | Shape |
        |---|---|
        | `mlp_attr` | `(N, L)` |
        """
    )
    st.markdown("**Hook target:** `mlp.c_proj.register_forward_hook` (captures output)")
    st.code(
        "# C — MLP attribution hook\n"
        "def _mlp_out(m, inp, out, buf=mlp_outputs):\n"
        "    buf.append(out[0, -1, :].detach().cpu())  # (hidden,)\n"
        "block.mlp.c_proj.register_forward_hook(_mlp_out)\n\n"
        "# Post-hoc: dot with W_U\n"
        "mlp_attr = [w_u @ m for m in mlp_outputs]     # scalar per layer",
        language="python",
    )

# --- D: MLP Neurons ---
with st.container(border=True):
    st.subheader("D -- MLP Pre-Activation Neurons")
    st.markdown(
        """
        **What it captures:** The output of `c_fc` (the MLP up-projection)
        *before* AbsTopK zeroes most neurons. This gives the full pre-sparsity
        activation pattern across all 8 192 neurons per layer.

        | | Shape | Notes |
        |---|---|---|
        | `mlp_neurons` | `(N, L, mlp_dim)` | Sparse -- mostly zeros after AbsTopK |
        """
    )
    st.markdown("**Hook target:** `mlp.c_fc.register_forward_hook`")
    st.code(
        "# D — MLP neuron hook\n"
        "def _mlp_fc(m, inp, out, buf=mlp_pre):\n"
        "    buf.append(out[0, -1, :].detach().cpu())  # (mlp_dim,)\n"
        "block.mlp.c_fc.register_forward_hook(_mlp_fc)",
        language="python",
    )

# --- E: Edge Graph ---
with st.container(border=True):
    st.subheader("E -- Edge Connectivity Graph")
    st.markdown(
        """
        **What it captures:** A static weight-sparsity graph built by
        inspecting every weight matrix. Only non-zero weights
        (|w| > epsilon) become edges. Extracted **once** per model -- no
        hook needed.

        Node types: `embedding`, `attn_q/k/v`, `attn_out`, `mlp_fc`,
        `mlp_proj`, `lm_head`.

        | | Format |
        |---|---|
        | `edge_graph.npz` | Dict of arrays: `src_name`, `dst_name`, `src_idx`, `dst_idx`, `weight`, `layer`, `param_key` |
        """
    )
    st.markdown("**Method:** Direct parameter inspection (no forward hook)")
    st.code(
        "# E — edge graph (sketch)\n"
        "for layer_idx, block in enumerate(transformer.h):\n"
        "    W = block.attn.c_proj.weight.detach().cpu().numpy()\n"
        "    nz_dst, nz_src = np.nonzero(np.abs(W) > eps)\n"
        "    # record (src, dst, weight, layer) for each non-zero entry\n"
        "    ...",
        language="python",
    )

st.divider()

# ── Output artifacts ─────────────────────────────────────────────────────────

st.header("Output Artifacts")

st.markdown(
    """
    All files are written to the same directory as the input, prefixed with
    `01_<stem>_` (e.g. `01_contrastive_stubs_residual_all.npy`).
    """
)

st.markdown(
    """
    | # | File | Format | Shape / Contents | Approx. Size |
    |---|------|--------|------------------|--------------|
    | 1 | `residual_final.npy` | float32 | `(N, H)` | ~70 MB |
    | 2 | `residual_all.npy` | float32 | `(N, L+1, H)` | ~600 MB |
    | 3 | `head_attr.npy` | float32 | `(N, L, n_heads)` | ~3 MB |
    | 4 | `mlp_attr.npy` | float32 | `(N, L)` | ~0.3 MB |
    | 5 | `mlp_neurons.npy` | float32 | `(N, L, mlp_dim)` | ~2.3 GB (sparse) |
    | 6 | `meta.json` | JSON | Prompt metadata (IDs, AST nodes, builtin objects, ...) | ~2 MB |
    | 7 | `edge_graph.npz` | NPZ dict | `src_name`, `dst_name`, `weight`, `layer`, ... | ~50 MB |
    """
)

st.divider()

# ── Downstream pipeline ─────────────────────────────────────────────────────

st.header("Downstream Pipeline")

st.markdown(
    """
    These artifacts feed directly into the remaining analysis steps:

    | Step | Script | Consumes | Purpose |
    |------|--------|----------|---------|
    | 02 | `02_variance_partition.py` | `residual_all`, `head_attr`, `mlp_attr`, `mlp_neurons`, `edge_graph` | Purity masks and variance partition across AST vs. builtin |
    | 03 | `03_rsa.py` | `residual_all`, `meta` | Representational Similarity Analysis (neural vs. model RDMs) |
    | 04 | `04_additivity.py` | `head_attr`, `mlp_attr`, `residual_all`, `meta` | Additivity scores and explicit-vs-proxy tests |
    | 05 | `05_probing.py` | `residual_all`, `head_attr`, `mlp_neurons`, `meta` | Linear probes and concept AUROC |
    | 06 | `06_ablation.py` | `head_attr`, `mlp_attr`, `edge_graph`, `meta` | Causal ablation and circuit identification |
    """
)
