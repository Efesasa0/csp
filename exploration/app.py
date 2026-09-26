"""
Streamlit presentation — AST x Builtin Mechanistic Interpretability
Start with: streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AST x Builtin — Contrastive Analysis",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
STUBS = ROOT / "contrastive_stubs.json"

# ─── Load dataset ────────────────────────────────────────────────────────────

@st.cache_data
def load_stubs() -> pd.DataFrame:
    rows = [json.loads(line) for line in open(STUBS)]
    return pd.DataFrame(rows)


df = load_stubs()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — Title & Motivation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.title("Does a Language Model Separate Syntax from Semantics?")
st.markdown(
    """
    We probe a **weight-sparse GPT-2** (OpenAI `circuit_sparsity`) with
    carefully constructed Python prompts to ask:

    > *Can we find independent subspaces in the residual stream —
    > one encoding **AST structure** (syntax) and another encoding
    > **builtin identity** (semantics)?*

    The first step is building a **contrastive dataset** that lets us
    tease apart these two factors.
    """
)

st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Dataset at a Glance
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.header("The Contrastive Stubs Dataset")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total prompts", f"{len(df):,}")
c2.metric("AST nodes", df["ast_node"].nunique())
c3.metric("Builtins", df["builtin_obj"].nunique())
c4.metric("Variant types", df["variant_type"].nunique())

st.markdown(
    """
    Each prompt is a short, syntactically valid Python snippet built from
    a **factorial design**: every AST construct (e.g. `For`, `While`, `If`)
    is crossed with every builtin (e.g. `range`, `len`, `sorted`).

    Variable names, structure, and style are held constant —
    the *only* things that change are the syntax node and the builtin call.
    """
)

st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — The Four Variant Types (the key design idea)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.header("Four Conditions, One Design")

st.markdown(
    """
    To isolate AST and builtin effects we need **baselines** — prompts
    where one factor is removed while the other is held constant.
    The dataset has four variant types:
    """
)

variant_info = {
    "explicit": (
        "AST + Builtin",
        "The full contrastive prompt — a real AST construct calling a real builtin.",
    ),
    "baseline_ast": (
        "AST only (no builtin)",
        "The AST construct appears with **no builtin call** — just raw data flow. "
        "Builtin is `__none__`. By comparing these activations against `explicit`, "
        "any difference is purely due to the builtin being present. "
        "Only 372 prompts: one set per AST node (no factorial crossing needed).",
    ),
    "baseline_builtin": (
        "Builtin only (no primary AST)",
        "The builtin appears in a minimal assignment — no `for`, `while`, etc. "
        "Any activation difference vs explicit is due to the AST node.",
    ),
    "proxy": (
        "AST + Builtin (indirect call)",
        "Same semantics, but the builtin is invoked indirectly "
        "(e.g. `data.__len__()` instead of `len(data)`). "
        "Tests whether the model relies on the *token* or the *concept*.",
    ),
}

for vtype, (short, explanation) in variant_info.items():
    subset = df[df["variant_type"] == vtype]
    count = len(subset)

    # Pick two diverse examples for explicit, one for the rest
    if vtype in ("explicit", "proxy"):
        samples = [
            subset.iloc[0],
            subset[subset["builtin_obj"] != subset.iloc[0]["builtin_obj"]].iloc[0],
        ]
    elif vtype == "baseline_ast":
        samples = [
            subset.iloc[0],
            subset[subset["ast_node"] != subset.iloc[0]["ast_node"]].iloc[0],
        ]
    else:
        samples = [subset.iloc[0]]

    with st.container(border=True):
        col_label, col_code = st.columns([1, 2])
        with col_label:
            st.subheader(vtype)
            st.caption(f"{short}  ·  {count:,} prompts")
            st.markdown(explanation)
        with col_code:
            for sample in samples:
                st.code(sample["prompt_text"], language="python")
                st.caption(
                    f"ast_node = `{sample['ast_node']}`  ·  "
                    f"builtin_obj = `{sample['builtin_obj']}`"
                )

st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Distributions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.header("Coverage")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("AST nodes")
    ast_counts = df["ast_node"].value_counts()
    st.bar_chart(ast_counts)

with col_b:
    st.subheader("Builtins")
    b_counts = df["builtin_obj"].value_counts()
    st.bar_chart(b_counts)

st.subheader("Variant type balance")
st.bar_chart(df["variant_type"].value_counts())

st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 5 — Browse the data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.header("Browse")

with st.expander("Filter & explore the full dataset"):
    fc1, fc2, fc3 = st.columns(3)
    sel_ast = fc1.multiselect("AST node", sorted(df["ast_node"].unique()))
    sel_b   = fc2.multiselect("Builtin", sorted(df["builtin_obj"].unique()))
    sel_v   = fc3.multiselect("Variant", sorted(df["variant_type"].unique()))

    filtered = df.copy()
    if sel_ast:
        filtered = filtered[filtered["ast_node"].isin(sel_ast)]
    if sel_b:
        filtered = filtered[filtered["builtin_obj"].isin(sel_b)]
    if sel_v:
        filtered = filtered[filtered["variant_type"].isin(sel_v)]

    st.dataframe(
        filtered[["prompt_id", "ast_node", "builtin_obj", "variant_type",
                  "prompt_text", "token_length"]],
        use_container_width=True,
        hide_index=True,
        height=400,
    )
    st.caption(f"Showing {len(filtered):,} of {len(df):,} prompts")
