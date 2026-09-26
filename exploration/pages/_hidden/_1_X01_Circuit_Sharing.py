import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="X01 — Circuit Sharing", layout="wide")

OUT = Path("experiments/outputs/x01")


def safe_image(path, caption=None):
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Not yet generated: {path.name}")


@st.cache_data
def load_data():
    return pd.read_csv(OUT / "within_vs_cross_group_jaccard.csv")


st.title("X01 — Circuit Sharing")
st.markdown(
    "Measures within-group vs. cross-group Jaccard similarity of neuron circuits "
    "across layers and concept groups (numeric, container)."
)

csv_path = OUT / "within_vs_cross_group_jaccard.csv"
if not csv_path.exists():
    st.warning("CSV not found — run experiment X01 first.")
    st.stop()

df = load_data()

# --- Sidebar filters ---
groups = sorted(df["group"].unique())
layers = sorted(df["layer"].unique())

st.sidebar.header("X01 Filters")
sel_groups = st.sidebar.multiselect("Groups", groups, default=groups)
layer_range = st.sidebar.slider(
    "Layer range", int(min(layers)), int(max(layers)), (int(min(layers)), int(max(layers)))
)

mask = df["group"].isin(sel_groups) & df["layer"].between(*layer_range)
filt = df[mask]

# --- Charts ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mean Jaccard by Layer & Scope")
    fig = px.bar(
        filt,
        x="layer",
        y="mean_jaccard",
        color="scope",
        barmode="group",
        facet_col="group",
        labels={"mean_jaccard": "Mean Jaccard", "layer": "Layer"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        "First plot shows that numeric and container style manual seperation gives no information, "
        "things are mostly similar in structure no matter this split."
    )

with col2:
    st.subheader("Within − Cross Jaccard Gap")
    within = filt[filt["scope"] == "within"][["group", "layer", "mean_jaccard"]].rename(
        columns={"group": "base_group", "mean_jaccard": "within"}
    )
    cross = filt[filt["scope"] == "cross"][["group", "layer", "mean_jaccard"]].copy()
    cross["base_group"] = cross["group"].str.replace("_vs_.*$", "", regex=True)
    cross = cross.rename(columns={"mean_jaccard": "cross"})

    gap_df = within.merge(cross[["base_group", "layer", "cross"]],
                          on=["base_group", "layer"], how="inner")
    if not gap_df.empty:
        gap_df["gap"] = gap_df["within"] - gap_df["cross"]
        fig2 = px.line(gap_df, x="layer", y="gap", color="base_group", markers=True,
                       labels={"gap": "Jaccard Gap (within − cross)", "layer": "Layer",
                               "base_group": "Group"})
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(
            "It implies there is almost no gap in Jaccard similarity. The results are around "
            "-0.006 to 0.010 and very small for both numeric and container gaps."
        )
    else:
        st.info("Need matched within/cross entries to compute gap.")

# --- PNGs ---
st.subheader("Heatmaps")
sel_layer = st.sidebar.selectbox("Heatmap layer", layers)
c1, c2 = st.columns(2)
with c1:
    safe_image(OUT / f"heatmap_numeric_layer{sel_layer}.png", f"Numeric — Layer {sel_layer}")
with c2:
    safe_image(OUT / f"heatmap_container_layer{sel_layer}.png", f"Container — Layer {sel_layer}")

# --- Table ---
st.subheader("Raw Data")
st.dataframe(filt, use_container_width=True)
