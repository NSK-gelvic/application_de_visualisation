# -*- coding: utf-8 -*-

import gc
import math

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from utils.db import get_conn
from modules.data_loader import load_geojson


st.set_page_config(layout="wide", page_title="Analyse Climat - Rendement")

st.title("Analyse Climat - Rendement")
st.markdown("---")


# =====================================================
# CONFIG
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

DISPLAY_LABELS = {
    "temp_moyenne": "Temperature moyenne (°C)",
    "precipitation_total": "Precipitations totales (mm)",
    "tmax_mean": "Temperature maximale moyenne (°C)",
    "tmin_mean": "Temperature minimale moyenne (°C)",
    "stress_climatique": "Stress climatique",
    "deficit_hydrique": "Deficit hydrique",
    "amplitude_thermique": "Amplitude thermique",
    "rendement": "Rendement (hl/ha)",
    "volume": "Volume (hl)",
    "zone": "Zone",
    "annee": "Annee",
    "code_couleur": "Couleur",
    "code_cepage": "Cepage",
    "code_departement": "Departement",
}

ZONE_COLOR_MAP = {
    "0": "#BDBDBD",
    "1": "#000000",
    "2": "#FF0000",
    "3": "#1A8F2A",
    "4": "#0033CC",
    "5": "#AFC6D9",
    "6": "#7A1FA2",
    "7": "#FFD800",
}

WINE_COLOR_MAP = {
    "BL": "#F4D03F",
    "RG": "#A93226",
    "RS": "#F1948A",
}


# =====================================================
# OUTILS
# =====================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def format_number(value, decimals: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.{decimals}f}"


def label_of(col: str) -> str:
    return DISPLAY_LABELS.get(col, col)


def minmax_scale(series: pd.Series) -> pd.Series:
    series = safe_numeric(series)
    if series.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    min_val = series.min()
    max_val = series.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series(1.0, index=series.index)
    return (series - min_val) / (max_val - min_val)


def inverse_minmax_scale(series: pd.Series) -> pd.Series:
    scaled = minmax_scale(series)
    return 1 - scaled


def build_zone_color_dict(zones) -> dict:
    result = {}
    for z in zones:
        z_str = str(int(z)) if pd.notna(z) else "0"
        result[z] = ZONE_COLOR_MAP.get(z_str, "#7F7F7F")
    return result


def class_from_score(score: float) -> str:
    if pd.isna(score):
        return "Non classee"
    if score >= 80:
        return "A"
    if score >= 65:
        return "B"
    if score >= 50:
        return "C"
    return "D"


def weighted_corr(df_in: pd.DataFrame, x: str, y: str) -> float:
    tmp = df_in[[x, y]].dropna().copy()
    if len(tmp) < 3:
        return np.nan
    return tmp[x].corr(tmp[y])


# =====================================================
# CHARGEMENT DES DONNEES
# =====================================================

@st.cache_data
def load_climate_yield_geo() -> pd.DataFrame:
    """Charge les données climat_rendement_geo"""
    conn = get_conn()
    try:
        df = conn.execute("SELECT * FROM climat_rendement_geo").df()
    finally:
        pass

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

    if "zone" in df.columns:
        df["zone"] = df["zone"].astype("Int64")

    if "annee" in df.columns:
        df["annee"] = df["annee"].astype("Int64")

    for col in ["commune", "code_departement"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


@st.cache_data
def load_fusion_analysis() -> pd.DataFrame:
    """Charge les données fusion"""
    conn = get_conn()
    try:
        df = conn.execute(
            """
            SELECT
                type_mvt,
                code_couleur,
                annee,
                volume,
                surface,
                code_cepage,
                cvi,
                rendement,
                commune,
                zone,
                code_departement,
                departement,
                Huglin_Index,
                Hot_D,
                Very_Hot_D,
                Climatic_Dryness_Index,
                temp_moyenne,
                precipitation_total
            FROM fusion
            """
        ).df()
    finally:
        pass

    df.columns = df.columns.astype(str).str.strip()

    numeric_cols = [
        "annee",
        "volume",
        "surface",
        "rendement",
        "zone",
        "Huglin_Index",
        "Hot_D",
        "Very_Hot_D",
        "Climatic_Dryness_Index",
        "temp_moyenne",
        "precipitation_total",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if "annee" in df.columns:
        df["annee"] = df["annee"].astype("Int64")

    if "zone" in df.columns:
        df["zone"] = df["zone"].astype("Int64")

    text_cols = [
        "type_mvt",
        "code_couleur",
        "code_cepage",
        "commune",
        "code_departement",
        "departement",
        "cvi",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    return df


# Chargement des données
with st.spinner("Chargement des donnees..."):
    df_geo = load_climate_yield_geo()
    df_fusion = load_fusion_analysis()

if df_geo.empty:
    st.warning("Aucune donnee climat_rendement_geo disponible.")
    st.stop()

if df_fusion.empty:
    st.warning("Aucune donnee fusion disponible.")
    st.stop()


# =====================================================
# SIDEBAR - FILTRES
# =====================================================

with st.sidebar:
    st.header("Filtres")
    st.divider()
    
    available_zones = sorted([int(z) for z in df_geo["zone"].dropna().unique()])
    selected_zones = st.multiselect(
        "Zones pedoclimatiques",
        available_zones,
        default=available_zones[:3] if len(available_zones) >= 3 else available_zones,
        help="Selectionnez une ou plusieurs zones a analyser"
    )

    available_deps = sorted(df_geo["code_departement"].dropna().astype(str).unique().tolist())
    selected_deps = st.multiselect(
        "Departements",
        available_deps,
        default=available_deps,
        help="Filtrer par departement"
    )

    available_years = sorted([int(y) for y in df_geo["annee"].dropna().unique()])
    
    col_year1, col_year2 = st.columns(2)
    with col_year1:
        year_min = st.number_input(
            "Annee min",
            min_value=int(min(available_years)),
            max_value=int(max(available_years)),
            value=int(min(available_years))
        )
    with col_year2:
        year_max = st.number_input(
            "Annee max",
            min_value=int(min(available_years)),
            max_value=int(max(available_years)),
            value=int(max(available_years))
        )
    
    selected_years = (year_min, year_max)

    available_colors = sorted([c for c in df_fusion["code_couleur"].dropna().unique() if c and c != "nan"])
    selected_colors = st.multiselect(
        "Couleurs",
        available_colors,
        default=available_colors,
        help="Type de vin : BL (Blanc), RG (Rouge), RS (Rose)"
    )

    available_cepages = sorted([c for c in df_fusion["code_cepage"].dropna().unique() if c and c != "nan"])
    selected_cepages = st.multiselect(
        "Cepages",
        available_cepages,
        default=[],
        help="Filtrer par cepage (optionnel)"
    )

    st.divider()
    
    use_last_5y = st.checkbox(
        "Focus sur les 5 dernieres annees",
        value=False,
        help="Limite l'analyse aux 5 dernieres annees de la periode selectionnee"
    )

if not selected_zones:
    st.warning("Selectionnez au moins une zone pour commencer l'analyse.")
    st.stop()

# Application des filtres
df_geo_filtered = df_geo[
    (df_geo["zone"].isin(selected_zones))
    & (df_geo["code_departement"].isin(selected_deps))
    & (df_geo["annee"].between(selected_years[0], selected_years[1]))
].copy()

df_fusion_filtered = df_fusion[
    (df_fusion["zone"].isin(selected_zones))
    & (df_fusion["code_departement"].isin(selected_deps))
    & (df_fusion["annee"].between(selected_years[0], selected_years[1]))
].copy()

if selected_colors:
    df_fusion_filtered = df_fusion_filtered[df_fusion_filtered["code_couleur"].isin(selected_colors)].copy()

if selected_cepages:
    df_fusion_filtered = df_fusion_filtered[df_fusion_filtered["code_cepage"].isin(selected_cepages)].copy()

if use_last_5y:
    max_year = int(df_geo_filtered["annee"].dropna().max())
    min_year_5 = max_year - 4
    df_geo_filtered = df_geo_filtered[df_geo_filtered["annee"].between(min_year_5, max_year)].copy()
    df_fusion_filtered = df_fusion_filtered[df_fusion_filtered["annee"].between(min_year_5, max_year)].copy()

if df_geo_filtered.empty or df_fusion_filtered.empty:
    st.warning("Aucune donnee disponible apres filtrage. Veuillez elargir vos criteres.")
    st.stop()


# =====================================================
# KPI CARDS
# =====================================================

st.header("Indicateurs cles")

col1, col2, col3, col4 = st.columns(4)

with col1:
    rendement_moy = df_geo_filtered["rendement"].mean()
    st.metric(
        "Rendement moyen",
        f"{format_number(rendement_moy, 1)} hl/ha",
        delta=None
    )

with col2:
    temp_moy = df_geo_filtered["temp_moyenne"].mean()
    st.metric(
        "Temperature moyenne",
        f"{format_number(temp_moy, 1)} °C",
        delta=None
    )

with col3:
    precip_moy = df_geo_filtered["precipitation_total"].mean()
    st.metric(
        "Precipitations moyennes",
        f"{format_number(precip_moy, 0)} mm",
        delta=None
    )

with col4:
    reve = df_fusion_filtered[df_fusion_filtered["type_mvt"] == "REVE"].copy()
    volume_total = reve["volume"].sum()
    st.metric(
        "Volume total",
        f"{format_number(volume_total, 0)} hl",
        delta=None
    )

st.markdown("---")


# =====================================================
# SCORING INTELLIGENT TYPE AOC
# =====================================================

with st.expander("Scoring intelligent des zones (type AOC)", expanded=True):
    zone_scoring = (
        df_geo_filtered.groupby("zone", as_index=False)
        .agg(
            rendement_moy=("rendement", "mean"),
            rendement_std=("rendement", "std"),
            temp_moy=("temp_moyenne", "mean"),
            precip_moy=("precipitation_total", "mean"),
            stress_moy=("stress_climatique", "mean"),
            deficit_moy=("deficit_hydrique", "mean"),
            amplitude_moy=("amplitude_thermique", "mean"),
        )
        .sort_values("zone")
    )

    # Climat de reference = zones les plus productives
    top_yield_threshold = zone_scoring["rendement_moy"].quantile(0.75) if len(zone_scoring) >= 4 else zone_scoring["rendement_moy"].median()
    top_zone_ref = zone_scoring[zone_scoring["rendement_moy"] >= top_yield_threshold].copy()

    if top_zone_ref.empty:
        ref_temp = zone_scoring["temp_moy"].median()
        ref_precip = zone_scoring["precip_moy"].median()
        ref_amp = zone_scoring["amplitude_moy"].median()
    else:
        ref_temp = top_zone_ref["temp_moy"].median()
        ref_precip = top_zone_ref["precip_moy"].median()
        ref_amp = top_zone_ref["amplitude_moy"].median()

    zone_scoring["score_rendement"] = minmax_scale(zone_scoring["rendement_moy"]) * 100
    zone_scoring["score_stabilite"] = inverse_minmax_scale(zone_scoring["rendement_std"].fillna(zone_scoring["rendement_std"].max())) * 100
    zone_scoring["score_stress"] = inverse_minmax_scale(zone_scoring["stress_moy"]) * 100
    zone_scoring["score_deficit"] = inverse_minmax_scale(zone_scoring["deficit_moy"]) * 100

    zone_scoring["score_temp_equilibre"] = (
        1 - (
            (zone_scoring["temp_moy"] - ref_temp).abs() /
            max((zone_scoring["temp_moy"] - ref_temp).abs().max(), 1e-9)
        )
    ) * 100

    zone_scoring["score_precip_equilibre"] = (
        1 - (
            (zone_scoring["precip_moy"] - ref_precip).abs() /
            max((zone_scoring["precip_moy"] - ref_precip).abs().max(), 1e-9)
        )
    ) * 100

    zone_scoring["score_amplitude_equilibre"] = (
        1 - (
            (zone_scoring["amplitude_moy"] - ref_amp).abs() /
            max((zone_scoring["amplitude_moy"] - ref_amp).abs().max(), 1e-9)
        )
    ) * 100

    for col in [
        "score_temp_equilibre",
        "score_precip_equilibre",
        "score_amplitude_equilibre",
    ]:
        zone_scoring[col] = zone_scoring[col].clip(lower=0, upper=100)

    zone_scoring["score_aoc"] = (
        0.35 * zone_scoring["score_rendement"]
        + 0.20 * zone_scoring["score_stabilite"]
        + 0.15 * zone_scoring["score_stress"]
        + 0.10 * zone_scoring["score_deficit"]
        + 0.10 * zone_scoring["score_temp_equilibre"]
        + 0.05 * zone_scoring["score_precip_equilibre"]
        + 0.05 * zone_scoring["score_amplitude_equilibre"]
    ).round(1)

    zone_scoring["classe_aoc"] = zone_scoring["score_aoc"].apply(class_from_score)
    zone_color_dict = build_zone_color_dict(zone_scoring["zone"].tolist())

    col_score1, col_score2 = st.columns([2, 1])
    
    with col_score1:
        fig_score = px.bar(
            zone_scoring.sort_values("score_aoc", ascending=False),
            x="zone",
            y="score_aoc",
            color="zone",
            color_discrete_map=zone_color_dict,
            text="classe_aoc",
            title="Classement qualitatif des zones",
            labels={"zone": "Zone", "score_aoc": "Score qualite"},
            height=500
        )
        fig_score.update_traces(textposition="outside", textfont_size=14)
        fig_score.update_layout(
            xaxis_type="category",
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(bgcolor="white", font_size=12)
        )
        st.plotly_chart(fig_score, key="score_bar_chart", use_container_width=True)

    with col_score2:
        st.markdown("### Detail des scores")
        st.dataframe(
            zone_scoring[
                [
                    "zone",
                    "classe_aoc",
                    "score_aoc",
                    "rendement_moy",
                    "rendement_std",
                ]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )


# =====================================================
# CARTE PAR ZONE
# =====================================================

with st.expander("Cartographie climat-rendement", expanded=True):
    try:
        geo_zones = load_geojson("zones")

        if geo_zones is None:
            st.error("Impossible de charger le GeoJSON des zones.")
        else:
            map_agg = (
                df_geo_filtered.groupby("zone", as_index=False)
                .agg(
                    rendement=("rendement", "mean"),
                    temp_moyenne=("temp_moyenne", "mean"),
                    precipitation_total=("precipitation_total", "mean"),
                    tmax_mean=("tmax_mean", "mean"),
                    tmin_mean=("tmin_mean", "mean"),
                    stress_climatique=("stress_climatique", "mean"),
                    deficit_hydrique=("deficit_hydrique", "mean"),
                    amplitude_thermique=("amplitude_thermique", "mean"),
                )
            )

            commune_zone = df_geo_filtered[["commune", "zone"]].drop_duplicates().copy()
            map_display = commune_zone.merge(map_agg, on="zone", how="inner")
            map_display["zone"] = map_display["zone"].astype(str)

            fig_map = px.choropleth(
                map_display,
                geojson=geo_zones,
                locations="commune",
                featureidkey="properties.code_commune",
                color="rendement",
                color_continuous_scale="YlOrRd",
                title="Rendement par zone",
                hover_data={
                    "zone": True,
                    "rendement": ":.1f",
                    "temp_moyenne": ":.1f",
                    "precipitation_total": ":.0f",
                    "tmax_mean": ":.1f",
                    "tmin_mean": ":.1f",
                    "stress_climatique": ":.2f",
                    "deficit_hydrique": ":.2f",
                    "amplitude_thermique": ":.1f",
                },
                labels={
                    "zone": "Zone",
                    "rendement": label_of("rendement"),
                },
                height=550
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(
                margin=dict(l=0, r=0, t=50, b=0),
                coloraxis_colorbar=dict(title="Rendement (hl/ha)", thickness=15)
            )
            st.plotly_chart(fig_map, key="climate_map", use_container_width=True)

    except Exception as e:
        st.warning(f"Carte indisponible : {e}")


# =====================================================
# EVOLUTION PAR ZONE
# =====================================================

st.header("Evolution des indicateurs par zone")

# Selecteurs pour l'evolution
col_evol1, col_evol2, col_evol3 = st.columns([1, 1, 1])

with col_evol1:
    evol_zone = st.multiselect(
        "Zones a comparer",
        options=selected_zones,
        default=selected_zones[:min(3, len(selected_zones))] if len(selected_zones) > 1 else selected_zones,
        help="Selectionnez les zones a afficher sur les graphiques"
    )

with col_evol2:
    evol_indicator = st.selectbox(
        "Indicateur a visualiser",
        options=["rendement"] + CLIMATE_VARS,
        format_func=label_of,
        index=0,
        key="evol_indicator"
    )

with col_evol3:
    focus_last_5y = st.checkbox(
        "Focus 5 dernieres annees",
        value=False,
        key="focus_last_5y",
        help="Limiter l'affichage aux 5 dernieres annees"
    )

if not evol_zone:
    st.info("Selectionnez au moins une zone pour visualiser l'evolution.")
else:
    # Preparation des donnees d'evolution
    evol_data = df_geo_filtered[df_geo_filtered["zone"].isin(evol_zone)].copy()
    
    if focus_last_5y:
        max_year_evol = evol_data["annee"].max()
        min_year_evol = max_year_evol - 4
        evol_data = evol_data[evol_data["annee"].between(min_year_evol, max_year_evol)].copy()
    
    evol_agg = (
        evol_data.groupby(["zone", "annee"], as_index=False)[evol_indicator]
        .mean()
        .sort_values(["zone", "annee"])
    )
    
    # Graphique d'evolution avec Plotly
    zone_colors = build_zone_color_dict(evol_zone)
    
    fig_evolution = go.Figure()
    
    for zone in evol_zone:
        zone_data = evol_agg[evol_agg["zone"] == zone].copy()
        if not zone_data.empty:
            color = zone_colors.get(zone, "#7F7F7F")
            fig_evolution.add_trace(go.Scatter(
                x=zone_data["annee"],
                y=zone_data[evol_indicator],
                mode="lines+markers",
                name=f"Zone {zone}",
                line=dict(width=3, color=color),
                marker=dict(size=8, color=color),
                hovertemplate=f"Zone {zone}<br>Annee: %{{x}}<br>{label_of(evol_indicator)}: %{{y:.1f}}<extra></extra>"
            ))
    
    fig_evolution.update_layout(
        title=f"Evolution de {label_of(evol_indicator)} par zone",
        xaxis_title="Annee",
        yaxis_title=label_of(evol_indicator),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray")
    )
    
    st.plotly_chart(fig_evolution, key="evolution_chart", use_container_width=True)
    
    # Tableau recapitulatif par zone
    with st.expander("Tableau recapitulatif par zone", expanded=False):
        recap = evol_data.groupby("zone")[evol_indicator].agg(["mean", "std", "min", "max"]).round(2).reset_index()
        recap.columns = ["zone", "Moyenne", "Ecart-type", "Minimum", "Maximum"]
        
        st.dataframe(recap, use_container_width=True, hide_index=True)


# =====================================================
# HISTOGRAMMES DE DISTRIBUTION
# =====================================================

with st.expander("Distributions des indicateurs", expanded=False):
    col_hist1, col_hist2 = st.columns(2)
    
    with col_hist1:
        hist_var = st.selectbox(
            "Variable a analyser",
            ["rendement"] + CLIMATE_VARS,
            format_func=label_of,
            key="hist_var_select",
        )
    
    with col_hist2:
        hist_group = st.radio(
            "Grouper par",
            ["Zone", "Couleur"],
            horizontal=True,
            key="hist_group_radio",
        )
    
    group_col = "zone" if hist_group == "Zone" else "code_couleur"
    
    if hist_group == "Zone":
        hist_source = df_geo_filtered[[hist_var, group_col]].dropna().copy()
    else:
        hist_source = df_fusion_filtered[[hist_var, group_col]].dropna().copy()
    
    if not hist_source.empty:
        fig_hist = go.Figure()
        
        groups = hist_source[group_col].dropna().unique()
        
        for grp in groups:
            data = hist_source[hist_source[group_col] == grp][hist_var].dropna()
            if not data.empty:
                if hist_group == "Zone":
                    color = ZONE_COLOR_MAP.get(str(int(grp)), "#7F7F7F")
                    name = f"Zone {grp}"
                else:
                    color = WINE_COLOR_MAP.get(str(grp), "#7F7F7F")
                    name = f"{grp}"
                
                fig_hist.add_trace(go.Histogram(
                    x=data,
                    name=name,
                    marker_color=color,
                    opacity=0.6,
                    nbinsx=20,
                    hovertemplate=f"{name}<br>Valeur: %{{x:.1f}}<br>Frequence: %{{y}}<extra></extra>"
                ))
        
        fig_hist.update_layout(
            title=f"Distribution de {label_of(hist_var)}",
            xaxis_title=label_of(hist_var),
            yaxis_title="Frequence",
            barmode="overlay",
            plot_bgcolor="rgba(0,0,0,0)",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray")
        )
        
        st.plotly_chart(fig_hist, key="histogram_chart", use_container_width=True)
    else:
        st.info("Donnees insuffisantes pour l'histogramme.")


# =====================================================
# INFLUENCE POSITIVE / NEGATIVE SUR RENDEMENT ET VOLUME
# =====================================================

with st.expander("Indicateurs influencant rendement et volume", expanded=False):
    analysis_scope = st.radio(
        "Portee de l'analyse",
        ["Toutes les annees filtrees", "5 dernieres annees"],
        horizontal=True,
        key="analysis_scope_radio"
    )
    
    if analysis_scope == "5 dernieres annees":
        recent_max = int(df_fusion_filtered["annee"].dropna().max())
        recent_min = recent_max - 4
        fusion_scope = df_fusion_filtered[df_fusion_filtered["annee"].between(recent_min, recent_max)].copy()
        geo_scope = df_geo_filtered[df_geo_filtered["annee"].between(recent_min, recent_max)].copy()
    else:
        fusion_scope = df_fusion_filtered.copy()
        geo_scope = df_geo_filtered.copy()
    
    # Correlations rendement
    corr_rows = []
    for var in CLIMATE_VARS:
        if var in geo_scope.columns:
            corr_rows.append({
                "indicateur": var,
                "corr_rendement": weighted_corr(geo_scope, var, "rendement"),
            })
    
    corr_table = pd.DataFrame(corr_rows)
    corr_table["effet_rendement"] = np.where(corr_table["corr_rendement"] >= 0, "positif (+)", "negatif (-)")
    
    # Correlations volume
    reve_scope = fusion_scope[fusion_scope["type_mvt"] == "REVE"].copy()
    volume_corr_rows = []
    for var in ["temp_moyenne", "precipitation_total", "Hot_D", "Very_Hot_D", "Huglin_Index", "Climatic_Dryness_Index"]:
        if var in reve_scope.columns:
            volume_corr_rows.append({
                "indicateur": var,
                "corr_volume": weighted_corr(reve_scope, var, "volume"),
            })
    
    volume_corr_table = pd.DataFrame(volume_corr_rows)
    volume_corr_table["effet_volume"] = np.where(volume_corr_table["corr_volume"] >= 0, "positif (+)", "negatif (-)")
    
    col_corr1, col_corr2 = st.columns(2)
    
    with col_corr1:
        st.subheader("Impact sur le rendement")
        if not corr_table.empty:
            fig_corr_rdt = px.bar(
                corr_table.sort_values("corr_rendement"),
                x="indicateur",
                y="corr_rendement",
                color="effet_rendement",
                color_discrete_map={"positif (+)": "#1A8F2A", "negatif (-)": "#C00000"},
                labels={"indicateur": "Indicateur", "corr_rendement": "Correlation"},
                title="Correlations avec le rendement",
                height=400
            )
            fig_corr_rdt.update_layout(showlegend=False)
            st.plotly_chart(fig_corr_rdt, key="corr_rendement_chart", use_container_width=True)
            st.dataframe(corr_table.round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Donnees insuffisantes pour l'analyse rendement.")
    
    with col_corr2:
        st.subheader("Impact sur le volume")
        if not volume_corr_table.empty:
            fig_corr_vol = px.bar(
                volume_corr_table.sort_values("corr_volume"),
                x="indicateur",
                y="corr_volume",
                color="effet_volume",
                color_discrete_map={"positif (+)": "#1A8F2A", "negatif (-)": "#C00000"},
                labels={"indicateur": "Indicateur", "corr_volume": "Correlation"},
                title="Correlations avec le volume",
                height=400
            )
            fig_corr_vol.update_layout(showlegend=False)
            st.plotly_chart(fig_corr_vol, key="corr_volume_chart", use_container_width=True)
            st.dataframe(volume_corr_table.round(3), use_container_width=True, hide_index=True)
        else:
            st.info("Donnees insuffisantes pour l'analyse volume.")


# =====================================================
# ANALYSE PAR COULEUR ET PAR CEPAGE
# =====================================================

with st.expander("Analyse par couleur et par cepage", expanded=False):
    subtab1, subtab2 = st.tabs(["Analyse par couleur", "Analyse par cepage"])
    
    with subtab1:
        color_scope = fusion_scope[fusion_scope["type_mvt"] == "DECR"].copy()
        
        if color_scope.empty or color_scope["code_couleur"].dropna().empty:
            st.info("Aucune donnee couleur disponible.")
        else:
            color_zone = (
                color_scope.groupby(["zone", "code_couleur"], as_index=False)
                .agg(rendement=("rendement", "mean"))
            )
            
            fig_color = px.bar(
                color_zone,
                x="zone",
                y="rendement",
                color="code_couleur",
                barmode="group",
                color_discrete_map=WINE_COLOR_MAP,
                title="Rendement moyen par zone et par couleur",
                labels={"zone": "Zone", "rendement": "Rendement (hl/ha)", "code_couleur": "Couleur"},
                height=450
            )
            st.plotly_chart(fig_color, key="color_bar_chart", use_container_width=True)
            
            color_reve = fusion_scope[fusion_scope["type_mvt"] == "REVE"].copy()
            if not color_reve.empty:
                color_ratio = (
                    color_reve.groupby("code_couleur", as_index=False)["volume"]
                    .sum()
                    .sort_values("volume", ascending=False)
                )
                total_volume = color_ratio["volume"].sum()
                color_ratio["ratio_volume_pct"] = np.where(
                    total_volume > 0,
                    100 * color_ratio["volume"] / total_volume,
                    np.nan,
                )
                
                fig_color_ratio = px.pie(
                    color_ratio,
                    values="ratio_volume_pct",
                    names="code_couleur",
                    color="code_couleur",
                    color_discrete_map=WINE_COLOR_MAP,
                    title="Poids relatif de chaque couleur dans le volume total",
                    hole=0.4,
                    height=450
                )
                st.plotly_chart(fig_color_ratio, key="color_ratio_chart", use_container_width=True)
    
    with subtab2:
        cepage_scope = fusion_scope[fusion_scope["type_mvt"] == "REVE"].copy()
        
        if cepage_scope.empty or cepage_scope["code_cepage"].dropna().empty:
            st.info("Aucune donnee cepage disponible.")
        else:
            top_cepages = (
                cepage_scope.groupby("code_cepage", as_index=False)["volume"]
                .sum()
                .sort_values("volume", ascending=False)
                .head(12)["code_cepage"]
                .tolist()
            )
            
            cepage_scope = cepage_scope[cepage_scope["code_cepage"].isin(top_cepages)].copy()
            
            cepage_zone_year = (
                cepage_scope.groupby(["annee", "zone", "code_cepage"], as_index=False)["volume"]
                .sum()
            )
            
            fig_cepage = px.bar(
                cepage_zone_year,
                x="annee",
                y="volume",
                color="code_cepage",
                barmode="group",
                facet_row="zone" if len(selected_zones) <= 4 else None,
                title="Evolution des cepages par zone",
                labels={"volume": "Volume (hl)", "annee": "Annee", "code_cepage": "Cepage"},
                height=500
            )
            st.plotly_chart(fig_cepage, key="cepage_evolution_chart", use_container_width=True)


# =====================================================
# TABLEAU DE SYNTHESE
# =====================================================

with st.expander("Tableau de synthese par zone et annee", expanded=False):
    table_zone_year = (
        df_geo_filtered.groupby(["zone", "annee"], as_index=False)
        .agg(
            rendement=("rendement", "mean"),
            temp_moyenne=("temp_moyenne", "mean"),
            precipitation_total=("precipitation_total", "mean"),
            tmax_mean=("tmax_mean", "mean"),
            tmin_mean=("tmin_mean", "mean"),
            stress_climatique=("stress_climatique", "mean"),
            deficit_hydrique=("deficit_hydrique", "mean"),
            amplitude_thermique=("amplitude_thermique", "mean"),
        )
        .sort_values(["zone", "annee"])
    )
    
    st.dataframe(table_zone_year.round(2), use_container_width=True, hide_index=True)


# =====================================================
# NARRATION AUTOMATIQUE
# =====================================================

with st.expander("Analyse automatique", expanded=False):
    if not zone_scoring.empty:
        top_zone = zone_scoring.sort_values("score_aoc", ascending=False).iloc[0]
        bottom_zone = zone_scoring.sort_values("score_aoc", ascending=True).iloc[0]
        most_stable = zone_scoring.sort_values("rendement_std", ascending=True).iloc[0]
        best_yield = zone_scoring.sort_values("rendement_moy", ascending=False).iloc[0]
        
        # Meilleure couleur
        best_color_text = "Information indisponible"
        color_perf = (
            fusion_scope[fusion_scope["type_mvt"] == "DECR"]
            .groupby("code_couleur", as_index=False)["rendement"]
            .mean()
            .sort_values("rendement", ascending=False)
        )
        if not color_perf.empty:
            row = color_perf.iloc[0]
            best_color_text = f"La couleur la plus performante est {row['code_couleur']} avec un rendement moyen de {row['rendement']:.1f} hl/ha."
        
        # Meilleur cepage
        best_cepage_text = "Information indisponible"
        cepage_perf = (
            fusion_scope[fusion_scope["type_mvt"] == "REVE"]
            .groupby("code_cepage", as_index=False)["volume"]
            .sum()
            .sort_values("volume", ascending=False)
        )
        if not cepage_perf.empty:
            row = cepage_perf.iloc[0]
            best_cepage_text = f"Le cepage le plus present est {row['code_cepage']} avec {row['volume']:.0f} hl."
        
        narrative = f"""
        ### Synthese de l'analyse
        
        Sur la periode analysee :
        
        - **Zone la mieux classee** : Zone {int(top_zone['zone'])} avec un score qualite de {top_zone['score_aoc']:.1f} (classe {top_zone['classe_aoc']})
        - **Zone la plus productive** : Zone {int(best_yield['zone'])} avec {best_yield['rendement_moy']:.2f} hl/ha
        - **Zone la plus stable** : Zone {int(most_stable['zone'])} (ecart-type de {most_stable['rendement_std']:.2f})
        
        {best_color_text}
        {best_cepage_text}
        """
        
        st.markdown(narrative)
        
        # Points d'attention
        st.subheader("Points d'attention")
        
        attention_points = []
        
        if pd.notna(top_zone["stress_moy"]) and top_zone["stress_moy"] > zone_scoring["stress_moy"].median():
            attention_points.append(f"La zone {int(top_zone['zone'])} presente un stress climatique superieur a la mediane.")
        
        if pd.notna(bottom_zone["deficit_moy"]) and bottom_zone["deficit_moy"] > zone_scoring["deficit_moy"].median():
            attention_points.append(f"La zone {int(bottom_zone['zone'])} souffre d'un deficit hydrique important.")
        
        if not volume_corr_table.empty:
            volume_valid = volume_corr_table.dropna(subset=["corr_volume"]).copy()
            if not volume_valid.empty:
                worst_vol = volume_valid.sort_values("corr_volume", ascending=True).iloc[0]
                attention_points.append(f"L'indicateur {worst_vol['indicateur']} est le plus defavorable pour le volume (correlation = {worst_vol['corr_volume']:.2f}).")
        
        if not attention_points:
            attention_points.append("Aucun signal critique fort n'a ete detecte sur le perimetre selectionne.")
        
        for point in attention_points:
            st.write(f"- {point}")

st.markdown("---")
st.caption(f"Analyse mise a jour le {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")


gc.collect()