"""
Model Details
Visual overview of the OpenAI circuit-sparsity GPT-2 model skeleton.
"""
import streamlit as st

st.set_page_config(page_title="Model Details", layout="wide")

st.title("Model Details")

st.markdown(
    """
    A **weight-sparse GPT-2 variant** with 8 transformer blocks,
    designed for mechanistic interpretability.  Four architectural
    departures from vanilla GPT-2 make circuits easier to isolate.
    """
)

# ── Key numbers ─────────────────────────────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Layers", 8)
c2.metric("Heads / layer", 12)
c3.metric("d_model", "2 048")
c4.metric("d_mlp", "8 192")
c5.metric("Vocab", "50 257")

st.divider()

# ── Full skeleton diagram ───────────────────────────────────────────────────

st.header("Transformer Skeleton")

st.graphviz_chart(
    """
    digraph model {
        rankdir=BT;
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", fontname="Helvetica",
              fontsize=11, margin="0.15,0.08"];
        edge [color="#666666", arrowsize=0.7];

        /* ── input ─────────────────────────────── */
        tokens [label="Input Tokens", fillcolor="#e8e8e8", shape=plaintext,
                fontsize=12, fontcolor="#333333"];
        wte    [label="Token Embedding\\nwte  (50257 × 2048)", fillcolor="#d4e6f1"];

        tokens -> wte;

        /* ── bigram shortcut ──────────────────── */
        bigram [label="Bigram Table\\n(50257 × 50257)\\nlearned logit bias",
                fillcolor="#fadbd8", fontcolor="#922b21"];
        tokens -> bigram [style=dashed, color="#c0392b",
                          label="  shortcut", fontname="Helvetica",
                          fontsize=9, fontcolor="#c0392b"];

        /* ── transformer blocks ───────────────── */

        subgraph cluster_blocks {
            label="× 8  Transformer Blocks";
            labeljust=l;
            fontname="Helvetica"; fontsize=12;
            style="dashed"; color="#888888";
            margin=16;

            /* residual in */
            res_in  [label="Residual Stream\\n(batch, seq, 2048)",
                     fillcolor="#eafaf1", shape=plaintext,
                     fontsize=10, fontcolor="#1e8449"];

            /* RMSNorm 1 */
            rn1 [label="RMSNorm₁", fillcolor="#fdebd0", fontcolor="#b9770e"];

            /* cat pos emb */
            cat1 [label="⊕ cat_pos_emb\\n(concat, not add)",
                  fillcolor="#fdebd0", fontcolor="#b9770e"];

            /* Attention */
            attn [label="Multi-Head Attention\\n12 heads  ·  d_head ≈ 171\\n\\nQ K V → c_attn\\nout   → c_proj",
                  fillcolor="#d5f5e3"];

            /* add 1 */
            add1 [label="+", shape=circle, width=0.3,
                  fillcolor="#eafaf1", fontsize=14];

            /* RMSNorm 2 */
            rn2 [label="RMSNorm₂", fillcolor="#fdebd0", fontcolor="#b9770e"];

            /* cat pos emb 2 */
            cat2 [label="⊕ cat_pos_emb\\n(concat, not add)",
                  fillcolor="#fdebd0", fontcolor="#b9770e"];

            /* MLP */
            mlp [label="MLP\\nc_fc   (2048 → 8192)\\nAbsTopK  (sparse act.)\\nc_proj (8192 → 2048)",
                 fillcolor="#d5f5e3"];

            /* add 2 */
            add2 [label="+", shape=circle, width=0.3,
                  fillcolor="#eafaf1", fontsize=14];

            /* residual out */
            res_out [label="Residual Stream (out)",
                     fillcolor="#eafaf1", shape=plaintext,
                     fontsize=10, fontcolor="#1e8449"];

            /* edges inside block */
            res_in -> rn1;
            rn1 -> cat1;
            cat1 -> attn;
            attn -> add1;
            res_in -> add1 [style=dashed, label="skip", fontsize=9,
                            fontname="Helvetica"];
            add1 -> rn2;
            rn2 -> cat2;
            cat2 -> mlp;
            mlp -> add2;
            add1 -> add2 [style=dashed, label="skip", fontsize=9,
                          fontname="Helvetica"];
            add2 -> res_out;
        }

        wte -> res_in;

        /* ── output head ──────────────────────── */
        ln_f    [label="RMSNorm (final)\\nln_f", fillcolor="#fdebd0",
                 fontcolor="#b9770e"];
        lm_head [label="Unembedding\\nlm_head  (2048 → 50257)",
                 fillcolor="#d4e6f1"];
        logits  [label="+ Logits", shape=circle, width=0.35,
                 fillcolor="#e8e8e8", fontsize=11];
        output  [label="Output Distribution", fillcolor="#e8e8e8",
                 shape=plaintext, fontsize=12, fontcolor="#333333"];

        res_out -> ln_f;
        ln_f -> lm_head;
        lm_head -> logits;
        bigram -> logits [style=dashed, color="#c0392b"];
        logits -> output;
    }
    """,
    use_container_width=True,
)

st.divider()

# ── What makes it different ─────────────────────────────────────────────────

st.header("Four Departures from Vanilla GPT-2")

dep1, dep2 = st.columns(2)

with dep1:
    with st.container(border=True):
        st.subheader("AbsTopK Activation")
        st.markdown(
            """
            Standard GPT-2 uses **GELU** in the MLP.  This model uses
            **AbsTopK**: after the up-projection (`c_fc`), only the
            top-K neurons by absolute value survive — the rest are
            zeroed out.

            The result: MLP layers are **weight-sparse feature
            detectors**, not dense transformations.  Circuits are
            cleaner and easier to attribute.
            """
        )
        st.code(
            "# Conceptual AbsTopK\n"
            "h = W_fc @ x            # (8192,)\n"
            "mask = topk(|h|, k)     # keep top-K\n"
            "h = h * mask            # zero the rest\n"
            "out = W_proj @ h        # (2048,)",
            language="python",
        )

with dep2:
    with st.container(border=True):
        st.subheader("RMSNorm (not LayerNorm)")
        st.markdown(
            """
            Every sub-layer uses **RMSNorm** instead of LayerNorm.
            RMSNorm rescales by the root-mean-square of activations
            *without* learning a bias or subtracting the mean.

            This means normalization **rescales magnitude** but does
            not add information — the directional content of the
            residual stream is preserved.
            """
        )
        st.code(
            "# RMSNorm vs LayerNorm\n"
            "# LayerNorm: (x - mean) / std * gamma + beta\n"
            "# RMSNorm:    x / RMS(x) * gamma\n"
            "#\n"
            "# RMS(x) = sqrt(mean(x^2))\n"
            "# No learned bias, no mean subtraction.",
            language="python",
        )

dep3, dep4 = st.columns(2)

with dep3:
    with st.container(border=True):
        st.subheader("Concatenated Position Embeddings")
        st.markdown(
            """
            Vanilla GPT-2 **adds** position embeddings to the token
            embeddings in the residual stream.  This model
            **concatenates** them inside each block before attention
            and MLP, then discards them.

            The residual stream stays **free of positional
            contamination** — a cleaner canvas for syntax/semantics
            factorisation.
            """
        )
        st.code(
            "# Vanilla GPT-2\n"
            "x = wte(tokens) + wpe(positions)   # mixed\n\n"
            "# circuit-sparsity\n"
            "x = wte(tokens)                     # clean\n"
            "# inside each block:\n"
            "h = concat(x, pos_emb)              # temp\n"
            "h = attention(h)  or  mlp(h)\n"
            "x = x + h[:, :d_model]              # pos discarded",
            language="python",
        )

with dep4:
    with st.container(border=True):
        st.subheader("Bigram Table")
        st.markdown(
            """
            A learned **(50 257 × 50 257)** logit bias matrix is added
            to the final logits, **bypassing the transformer entirely**.

            It encodes "token X was just seen → bias toward token Y".
            This offloads simple co-occurrence statistics from the
            transformer, letting the 8 blocks focus on higher-order
            structure.

            For our analysis, the bigram contribution must be
            **subtracted** to isolate the transformer's own circuits.
            """
        )
        st.code(
            "# Final logit computation\n"
            "transformer_logits = lm_head(ln_f(residual))\n"
            "bigram_bias = bigram_table[last_token]\n"
            "logits = transformer_logits + bigram_bias",
            language="python",
        )
