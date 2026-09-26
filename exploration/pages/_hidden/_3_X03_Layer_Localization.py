import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="X03 — Layer Localization", layout="wide")

OUT = Path("experiments/outputs/x03")


def safe_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not yet generated: {path.name}")


@st.cache_data
def load_data():
    return pd.read_csv(OUT / "localization_scores.csv")


st.title("X03 — Layer Localization")
st.markdown(
    "Measures how sharply each concept's neuron density is localized across layers "
    "using entropy and peak-layer analysis."
)

csv_path = OUT / "localization_scores.csv"
if not csv_path.exists():
    st.warning("CSV not found — run experiment X03 first.")
    st.stop()

df = load_data()

# --- Sidebar filters ---
kinds = sorted(df["kind"].unique())

st.sidebar.header("X03 Filters")
sel_kind = st.sidebar.radio("Kind", kinds)
concept_search = st.sidebar.text_input("Concept search")

mask = df["kind"] == sel_kind
if concept_search:
    mask = mask & df["concept"].str.contains(concept_search, case=False, na=False)
filt = df[mask]

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Density Heatmap")
    density_cols = [c for c in df.columns if c.startswith("density_L")]
    if len(filt) > 0:
        heat_df = filt.set_index("concept")[density_cols]
        fig = px.imshow(
            heat_df.values,
            x=[c.replace("density_", "") for c in density_cols],
            y=heat_df.index.tolist(),
            aspect="auto",
            color_continuous_scale="Viridis",
            labels={"x": "Layer", "y": "Concept", "color": "Density"},
        )
        fig.update_layout(height=max(400, len(filt) * 20))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No concepts match the current filter.")

with col2:
    st.subheader("Entropy vs Peak Layer")
    fig2 = px.scatter(
        filt,
        x="peak_layer",
        y="localization_entropy",
        color="kind",
        hover_data=["concept"],
        labels={"localization_entropy": "Localization Entropy", "peak_layer": "Peak Layer"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- PNGs ---
st.subheader("Static Plots")
c1, c2 = st.columns(2)
with c1:
    safe_image(OUT / f"concept_density_heatmap_{sel_kind}.png", f"Density Heatmap — {sel_kind}")
with c2:
    safe_image(OUT / f"stacked_bar_layer_density_{sel_kind}.png", f"Stacked Bar — {sel_kind}")

# --- Table ---
st.subheader("Raw Data")
st.dataframe(filt, use_container_width=True)
