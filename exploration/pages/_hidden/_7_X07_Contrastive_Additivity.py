import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="X07 — Contrastive Additivity", layout="wide")

OUT = Path("experiments/outputs/x07")


def safe_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not yet generated: {path.name}")


st.title("X07 — Contrastive Additivity")
st.markdown(
    "Tests whether AST and builtin representations are **additive** (independent) "
    "or **entangled** (interaction circuit) in the residual stream.\n\n"
    "For each (AST, builtin) pair:\n"
    "- `additivity = cos_sim(act_combined, act_ast_baseline + act_builtin_baseline)`\n"
    "- Score near **1.0** = independent/additive representations\n"
    "- Score near **0.0** = strong interaction circuit"
)

if not OUT.exists():
    st.warning("X07 output directory not found — run `bash run_experiments.sh --x07` first.")
    st.stop()

# ── Layer-level summary ──────────────────────────────────────────────────

layer_csv = OUT / "layer_additivity.csv"
if layer_csv.exists():
    st.subheader("Additivity by Layer")
    layer_df = pd.read_csv(layer_csv)
    fig = px.line(
        layer_df, x="layer", y="mean_additivity",
        markers=True,
        labels={"mean_additivity": "Mean Additivity Score", "layer": "Layer"},
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    if "std_additivity" in layer_df.columns:
        fig.add_scatter(
            x=layer_df["layer"], y=layer_df["mean_additivity"] + layer_df["std_additivity"],
            mode="lines", line=dict(width=0), showlegend=False,
        )
        fig.add_scatter(
            x=layer_df["layer"], y=layer_df["mean_additivity"] - layer_df["std_additivity"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(99, 110, 250, 0.2)", showlegend=False,
        )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(layer_df, use_container_width=True)

# ── Plots ────────────────────────────────────────────────────────────────

st.subheader("Additivity Heatmap")
st.markdown("Rows = AST nodes, columns = builtins. Green = additive, red = interaction circuit.")
safe_image(OUT / "additivity_heatmap.png")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Layer Traces")
    st.markdown("Top-10 most entangled (left) vs most additive (right) pairs across layers.")
    safe_image(OUT / "layer_traces.png")

with col2:
    st.subheader("Baseline Cosine Similarities")
    st.markdown("How orthogonal are AST / builtin directions? Blue = orthogonal, red = aligned.")
    safe_image(OUT / "baseline_cosine_matrices.png")

st.subheader("Explicit vs Proxy")
st.markdown(
    "Do proxy stubs (builtin token absent, e.g. `.__len__()` instead of `len()`) score "
    "similarly to explicit stubs? Similar = semantic encoding, gap = lexical encoding."
)
safe_image(OUT / "explicit_vs_proxy.png")

# ── Pair-level data ──────────────────────────────────────────────────────

pair_csv = OUT / "pair_additivity.csv"
if pair_csv.exists():
    st.subheader("Pair-Level Additivity Scores")
    pair_df = pd.read_csv(pair_csv)

    # Distribution
    fig_hist = px.histogram(
        pair_df, x="additivity_score", nbins=40,
        labels={"additivity_score": "Additivity Score"},
        title="Distribution of pair-level additivity scores",
    )
    fig_hist.add_vline(x=pair_df["additivity_score"].mean(), line_dash="dash",
                       annotation_text=f"mean={pair_df['additivity_score'].mean():.3f}")
    st.plotly_chart(fig_hist, use_container_width=True)

    # Top interactions / top additive
    col_a, col_b = st.columns(2)
    top_interact_csv = OUT / "top_interactions.csv"
    top_additive_csv = OUT / "top_additive.csv"

    with col_a:
        st.markdown("**Strongest interaction circuits** (lowest additivity)")
        if top_interact_csv.exists():
            st.dataframe(pd.read_csv(top_interact_csv), use_container_width=True)
        else:
            st.dataframe(
                pair_df.nsmallest(20, "additivity_score")[["pair", "additivity_score"]],
                use_container_width=True,
            )

    with col_b:
        st.markdown("**Most additive pairs** (highest additivity)")
        if top_additive_csv.exists():
            st.dataframe(pd.read_csv(top_additive_csv), use_container_width=True)
        else:
            st.dataframe(
                pair_df.nlargest(20, "additivity_score")[["pair", "additivity_score"]],
                use_container_width=True,
            )

    # Full table
    with st.expander("Full pair table"):
        st.dataframe(pair_df.sort_values("additivity_score"), use_container_width=True)
