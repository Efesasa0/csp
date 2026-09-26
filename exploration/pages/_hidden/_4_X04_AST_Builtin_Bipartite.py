import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="X04 — AST-Builtin Bipartite", layout="wide")

OUT = Path("experiments/outputs/x04")


def safe_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not yet generated: {path.name}")


@st.cache_data
def load_data():
    return pd.read_csv(OUT / "edge_list.csv")


st.title("X04 — AST-Builtin Bipartite")
st.markdown(
    "Weighted bipartite graph linking AST node types to built-in objects, based on "
    "shared neuron circuit overlap."
)

csv_path = OUT / "edge_list.csv"
if not csv_path.exists():
    st.warning("CSV not found — run experiment X04 first.")
    st.stop()

df = load_data()

# --- Sidebar filters ---
ast_families = sorted(df["ast_family"].unique())
builtin_domains = sorted(df["builtin_domain"].unique())

st.sidebar.header("X04 Filters")
sel_ast = st.sidebar.multiselect("AST Family", ast_families, default=ast_families)
sel_builtin = st.sidebar.multiselect("Builtin Domain", builtin_domains, default=builtin_domains)
w_min, w_max = float(df["weight"].min()), float(df["weight"].max())
min_weight = st.sidebar.slider("Min weight", w_min, w_max, w_min)

mask = df["ast_family"].isin(sel_ast) & df["builtin_domain"].isin(sel_builtin) & (df["weight"] >= min_weight)
filt = df[mask]

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top-20 Edges by Weight")
    top20 = filt.nlargest(20, "weight")
    top20 = top20.copy()
    top20["edge"] = top20["ast_node"] + " ↔ " + top20["builtin_obj"]
    fig = px.bar(
        top20.sort_values("weight"),
        x="weight",
        y="edge",
        orientation="h",
        color="ast_family",
        labels={"weight": "Weight", "edge": "Edge"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AST Node vs Builtin Object")
    fig2 = px.scatter(
        filt,
        x="ast_node",
        y="builtin_obj",
        size="weight",
        color="ast_family",
        hover_data=["weight", "builtin_domain"],
        labels={"ast_node": "AST Node", "builtin_obj": "Builtin Object"},
    )
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, use_container_width=True)

# --- PNGs ---
st.subheader("Static Plots")
c1, c2 = st.columns(2)
with c1:
    safe_image(OUT / "bipartite_graph.png", "Bipartite Graph")
with c2:
    safe_image(OUT / "ast_builtin_similarity_matrix.png", "Similarity Matrix")

# --- Table ---
st.subheader("Raw Data")
st.dataframe(filt, use_container_width=True)
