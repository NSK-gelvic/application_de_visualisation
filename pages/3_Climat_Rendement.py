# -*- coding: utf-8 -*-

import gc
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
import streamlit.components.v1 as components

from shapely.geometry import shape

from utils.db import get_connection

st.title("Analyse Climat ↔ Rendement")


# =====================================================
# UTILS
# =====================================================

CLIMATE_VARS = [
    "temp_moyenne",
    "precipitation_total",
    "tmax_mean",
    "tmin_mean",
    "stress_climatique",
    "deficit_hydrique",
    "amplitude_thermique",
]

LABELS = {
    "temp_moyenne": "Température moyenne (°C)",
    "precipitation_total": "Précipitations totales (mm)",
    "tmax_mean": "Température maximale moyenne (°C)",
    "tmin_mean": "Température minimale moyenne (°C)",
    "stress_climatique": "Stress climatique",
    "deficit_hydrique": "Déficit hydrique",
    "amplitude_thermique": "Amplitude thermique",
    "rendement": "Rendement",
}


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def format_number(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{decimals}f}"


def label_of(col: str) -> str:
    return LABELS.get(col, col)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data() -> pd.DataFrame:
    conn = get_connection(read_only=True)
    try:
        df = conn.execute("SELECT * FROM climat_rendement_geo").df()
    finally:
        conn.close()

    df.columns = df.columns.astype(str).str.strip()

    numeric_cols = [
        "zone",
        "annee",
        "temp_moyenne",
        "precipitation_total",
        "tmax_mean",
        "tmin_mean",
        "stress_climatique",
        "deficit_hydrique",
        "amplitude_thermique",
        "rendement",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if "annee" in df.columns:
        df["annee"] = df["annee"].astype("Int64")

    if "zone" in df.columns:
        df["zone"] = df["zone"].astype("Int64")

    if "code_departement" in df.columns:
        df["code_departement"] = df["code_departement"].astype(str).str.strip()

    if "commune" in df.columns:
        df["commune"] = df["commune"].astype(str).str.strip()

    return df


df = load_data()

if df.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Filtres")

zones = sorted([int(z) for z in df["zone"].dropna().unique()])
selected_zones = st.sidebar.multiselect(
    "Zones",
    zones,
    default=zones[:3] if len(zones) >= 3 else zones,
)

departements = sorted(df["code_departement"].dropna().astype(str).unique().tolist())
selected_deps = st.sidebar.multiselect(
    "Départements",
    departements,
    default=departements,
)

years = sorted([int(y) for y in df["annee"].dropna().unique()])
selected_years = st.sidebar.slider(
    "Années",
    min_value=int(min(years)),
    max_value=int(max(years)),
    value=(int(min(years)), int(max(years))),
)

if not selected_zones:
    st.warning("Sélectionne au moins une zone.")
    st.stop()


# =====================================================
# FILTER
# =====================================================

df_filtered = df[
    (df["zone"].isin(selected_zones))
    & (df["code_departement"].isin(selected_deps))
    & (df["annee"].between(selected_years[0], selected_years[1]))
].copy()

if df_filtered.empty:
    st.warning("Aucune donnée après filtrage.")
    st.stop()


# =====================================================
# KPI
# =====================================================

st.header("Indicateurs clés")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rendement moyen", format_number(df_filtered["rendement"].mean(), 1))

with col2:
    st.metric("Température moyenne", format_number(df_filtered["temp_moyenne"].mean(), 1))

with col3:
    st.metric("Précipitations moyennes", format_number(df_filtered["precipitation_total"].mean(), 0))


# =====================================================
# EVOLUTION TEMPORELLE
# =====================================================

st.header("Évolution climat vs rendement")

indicator = st.selectbox(
    "Indicateur climatique",
    CLIMATE_VARS,
    format_func=label_of,
)

df_grouped = (
    df_filtered.groupby("annee", as_index=False)
    .mean(numeric_only=True)
    .sort_values("annee")
)

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(
    df_grouped["annee"].astype(int),
    df_grouped["rendement"],
    marker="o",
    linewidth=2,
    label="Rendement",
)
ax1.set_ylabel("Rendement")
ax1.set_xlabel("Année")
ax1.set_xticks(df_grouped["annee"].astype(int))

ax2 = ax1.twinx()
ax2.plot(
    df_grouped["annee"].astype(int),
    df_grouped[indicator],
    color="orange",
    marker="o",
    linewidth=2,
    label=label_of(indicator),
)
ax2.set_ylabel(label_of(indicator))

ax1.set_title(f"Évolution du rendement vs {label_of(indicator)}")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

st.pyplot(fig)
plt.close(fig)


# =====================================================
# CORRELATION
# =====================================================

st.header("Corrélations climat → rendement")

corr_cols = ["rendement"] + [c for c in CLIMATE_VARS if c in df_filtered.columns]
corr_df = df_filtered[corr_cols].copy().dropna(how="all")

corr = corr_df.corr(numeric_only=True)

fig_corr, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, aspect="auto")

ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels([label_of(c) for c in corr.columns], rotation=45, ha="right")
ax.set_yticks(range(len(corr.index)))
ax.set_yticklabels([label_of(c) for c in corr.index])

for i in range(len(corr.index)):
    for j in range(len(corr.columns)):
        value = corr.iloc[i, j]
        txt = "NA" if pd.isna(value) else f"{value:.2f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9)

fig_corr.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Matrice de corrélation")
fig_corr.tight_layout()

st.pyplot(fig_corr)
plt.close(fig_corr)


# =====================================================
# SCATTER + REGRESSION
# =====================================================

st.header("Relation directe climat → rendement")

x_var = st.selectbox(
    "Variable explicative",
    CLIMATE_VARS,
    format_func=label_of,
    key="x_var_climat_rdt",
)

df_clean = df_filtered[[x_var, "rendement"]].dropna().copy()

fig_scatter, ax = plt.subplots(figsize=(8, 5))

ax.scatter(df_clean[x_var], df_clean["rendement"], alpha=0.5)

if len(df_clean) > 2:
    x = df_clean[x_var].to_numpy(dtype=float)
    y = df_clean["rendement"].to_numpy(dtype=float)

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    z = np.polyfit(x_sorted, y_sorted, 1)
    p = np.poly1d(z)
    ax.plot(x_sorted, p(x_sorted), linestyle="--", linewidth=2)

ax.set_xlabel(label_of(x_var))
ax.set_ylabel("Rendement")
ax.set_title(f"Impact de {label_of(x_var)} sur le rendement")
ax.grid(axis="y", linestyle="--", alpha=0.4)

st.pyplot(fig_scatter)
plt.close(fig_scatter)


# =====================================================
# PAR ZONE
# =====================================================

st.header("Rendement moyen par zone")

df_zone = (
    df_filtered.groupby("zone", as_index=False)["rendement"]
    .mean()
    .sort_values("zone")
)

fig_zone, ax = plt.subplots(figsize=(8, 4))
ax.bar(df_zone["zone"].astype(str), df_zone["rendement"])
ax.set_title("Rendement moyen par zone")
ax.set_xlabel("Zone")
ax.set_ylabel("Rendement")
ax.grid(axis="y", linestyle="--", alpha=0.4)

st.pyplot(fig_zone)
plt.close(fig_zone)


# =====================================================
# PAR DEPARTEMENT
# =====================================================

st.header("Rendement moyen par département")

df_dep = (
    df_filtered.groupby("code_departement", as_index=False)["rendement"]
    .mean()
    .sort_values("code_departement")
)

fig_dep, ax = plt.subplots(figsize=(10, 4))
ax.bar(df_dep["code_departement"], df_dep["rendement"])
ax.tick_params(axis="x", rotation=45)
ax.set_title("Rendement moyen par département")
ax.set_xlabel("Département")
ax.set_ylabel("Rendement")
ax.grid(axis="y", linestyle="--", alpha=0.4)

st.pyplot(fig_dep)
plt.close(fig_dep)


# =====================================================
# TABLEAU DETAILLE
# =====================================================

st.header("Tableau climat ↔ rendement par année")

group_mode = st.radio(
    "Regrouper par",
    ["zone", "code_departement"],
    horizontal=True,
)

table_cols = [
    group_mode,
    "annee",
    "rendement",
    "temp_moyenne",
    "precipitation_total",
    "tmax_mean",
    "tmin_mean",
    "stress_climatique",
    "deficit_hydrique",
    "amplitude_thermique",
]

available_table_cols = [c for c in table_cols if c in df_filtered.columns]

df_table = (
    df_filtered[available_table_cols]
    .groupby([group_mode, "annee"], as_index=False)
    .mean(numeric_only=True)
    .sort_values([group_mode, "annee"])
)

st.dataframe(df_table, use_container_width=True)


# =====================================================
# CARTE
# =====================================================

st.header("Carte rendement par zone")

try:
    df_map = df_filtered.copy()

    # conversion geometry JSON -> shapely
    df_map["geometry"] = df_map["geometry"].apply(
        lambda g: shape(g) if isinstance(g, dict) else None
    )

    gdf = gpd.GeoDataFrame(df_map, geometry="geometry")
    gdf = gdf.dropna(subset=["geometry"]).copy()

    if gdf.empty:
        st.warning("Aucune géométrie exploitable pour la carte.")
    else:
        gdf = gdf.dissolve(by="zone", aggfunc="mean", numeric_only=True).reset_index()

        center_lat = df_filtered["temp_moyenne"].notna().sum()
        _ = center_lat  # silence lint implicite

        m = folium.Map(location=[43.7, 3.5], zoom_start=8)

        rend_min = gdf["rendement"].min()
        rend_max = gdf["rendement"].max()

        def zone_color(value):
            if pd.isna(value):
                return "#D9D9D9"
            if rend_min == rend_max:
                return "#ff9999"
            ratio = (value - rend_min) / (rend_max - rend_min)
            if ratio < 0.25:
                return "#fee5d9"
            if ratio < 0.50:
                return "#fcae91"
            if ratio < 0.75:
                return "#fb6a4a"
            return "#cb181d"

        folium.GeoJson(
            gdf,
            style_function=lambda x: {
                "fillColor": zone_color(x["properties"].get("rendement")),
                "color": "black",
                "weight": 1,
                "fillOpacity": 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["zone", "rendement"],
                aliases=["Zone", "Rendement moyen"],
                localize=True,
            ),
        ).add_to(m)

        components.html(m._repr_html_(), height=600)

except Exception as e:
    st.warning(f"Carte indisponible : {e}")


# =====================================================
# INSIGHTS AUTO
# =====================================================

st.header("Insights automatiques")

corr_temp = corr.loc["rendement", "temp_moyenne"] if "temp_moyenne" in corr.columns else np.nan
corr_precip = (
    corr.loc["rendement", "precipitation_total"]
    if "precipitation_total" in corr.columns
    else np.nan
)

if pd.notna(corr_temp):
    if corr_temp > 0:
        st.success(f"Le rendement augmente avec la température (corr = {corr_temp:.2f})")
    else:
        st.error(f"Le rendement diminue avec la température (corr = {corr_temp:.2f})")
else:
    st.info("Corrélation température/rendement indisponible.")

if pd.notna(corr_precip):
    if corr_precip > 0:
        st.success(f"Plus de pluie → meilleur rendement (corr = {corr_precip:.2f})")
    else:
        st.warning(f"Trop de pluie peut réduire le rendement (corr = {corr_precip:.2f})")
else:
    st.info("Corrélation pluie/rendement indisponible.")


gc.collect()