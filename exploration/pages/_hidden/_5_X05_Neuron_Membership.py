import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="X05 — Neuron Membership", layout="wide")

OUT = Path("experiments/outputs/x05")


def safe_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not yet generated: {path.name}")


@st.cache_data
def load_data():
    return pd.read_csv(OUT / "top_neurons_per_layer.csv")


st.title("X05 — Neuron Membership")
st.markdown(
    "Identifies the most circuit-promiscuous neurons — those appearing in the highest "
    "number of concept circuits per layer."
)

csv_path = OUT / "top_neurons_per_layer.csv"
if not csv_path.exists():
    st.warning("CSV not found — run experiment X05 first.")
    st.stop()

df = load_data()

# --- Sidebar filters ---
layers = sorted(df["layer"].unique())

st.sidebar.header("X05 Filters")
sel_layer = st.sidebar.selectbox("Layer", layers)
top_n = st.sidebar.slider("Top-N neurons", 5, 20, 10)

layer_df = df[df["layer"] == sel_layer].nlargest(top_n, "n_circuits")

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Top-{top_n} Neurons — Layer {sel_layer}")
    layer_df = layer_df.copy()
    layer_df["neuron_label"] = layer_df["neuron_id"].astype(str)
    fig = px.bar(
        layer_df.sort_values("n_circuits"),
        x="n_circuits",
        y="neuron_label",
        orientation="h",
        labels={"n_circuits": "Circuit Count", "neuron_label": "Neuron ID"},
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Circuit Membership Distribution")
    fig2 = px.box(
        df,
        x="layer",
        y="n_circuits",
        labels={"n_circuits": "Circuit Count", "layer": "Layer"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- PNGs ---
st.subheader("Static Plots")
c1, c2 = st.columns(2)
with c1:
    safe_image(OUT / "neuron_membership_heatmap.png", "Membership Heatmap")
with c2:
    safe_image(OUT / "per_layer_histogram.png", "Per-Layer Histogram")

# --- Table ---
st.subheader("Raw Data")
st.dataframe(df, use_container_width=True)
