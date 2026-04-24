# -*- coding: utf-8 -*-

from pathlib import Path
import gc

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium
import geopandas as gpd
from branca.colormap import linear, LinearColormap
from scipy import stats
from scipy.signal import savgol_filter

# Import de la connexion partagée
from utils.db import get_conn

# =====================================================
# PATHS
# =====================================================

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_DIR

st.set_page_config(
    page_title="Modelisation climatique - Pays d'Oc",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Style CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #2c3e50;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .metric-label {
        color: #6c757d;
        font-size: 0.85rem;
    }
    hr {
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌡️ Modelisation climatique - Pays d'Oc</h1>
    <p>Analyse historique, tendances et projections climatiques des zones pedoclimatiques</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# CONSTANTES
# =====================================================

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

ZONE_LABELS = {
    "0": "Zone 0 : non classee / hors zonage",
    "1": "Zone 1 : zone humide de l'arriere-pays",
    "2": "Zone 2 : montagne, sols acides et peu profonds",
    "3": "Zone 3 : Piedmont, reserve utile limitee",
    "4": "Zone 4 : froide et seche autour du Pic Saint-Loup",
    "5": "Zone 5 : sols de qualite moyenne dans l'arriere-pays",
    "6": "Zone 6 : sols profonds, cotes temperees",
    "7": "Zone 7 : tres chaud, sols profonds",
}

SCENARIO_COLOR_MAP = {
    "optimiste": "#AFC6D9",
    "neutre": "#C99700",
    "pessimiste": "#C00000",
}

MAP_INDICATORS = [
    "temp_moyenne",
    "tmax_mean",
    "tmin_mean",
    "precipitation_total",
]

SCENARIO_INDICATORS = [
    "temp_moyenne",
    "tmax_mean",
    "tmin_mean",
    "precipitation_total",
]

HISTORICAL_INDICATORS = [
    "temp_moyenne",
    "tmax_mean",
    "tmin_mean",
    "precipitation_total",
    "Huglin_Index",
    "Hot_D",
    "Very_Hot_D",
    "Frost_D",
    "Late_Frost",
    "stress_climatique",
    "deficit_hydrique",
    "Climatic_Dryness_Index",
    "Soil_Water_Stock",
    "Soil_pH",
    "jours_secs",
    "jours_pluie",
]


# =====================================================
# FONCTIONS UTILITAIRES
# =====================================================

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    )


def indicator_label(indicator: str) -> str:
    labels = {
        "temp_moyenne": "Temperature moyenne (°C)",
        "tmax_mean": "Temperature maximale moyenne (°C)",
        "tmin_mean": "Temperature minimale moyenne (°C)",
        "precipitation_total": "Precipitations totales (mm)",
        "precipitation_total_avril_septembre": "Precipitations avril-septembre (mm)",
        "Huglin_Index": "Indice de Huglin",
        "Hot_D": "Nombre de jours de chaleur",
        "Very_Hot_D": "Nombre de jours de forte chaleur",
        "Frost_D": "Nombre de jours de gel",
        "Late_Frost": "Nombre de jours de gel tardif",
        "stress_climatique": "Stress climatique",
        "deficit_hydrique": "Deficit hydrique",
        "Climatic_Dryness_Index": "Indice de secheresse climatique",
        "Soil_Water_Stock": "Reserve utile en eau du sol",
        "Soil_pH": "pH du sol",
        "jours_secs": "Jours secs",
        "jours_pluie": "Jours de pluie",
    }
    return labels.get(indicator, indicator)


def format_value(value, indicator: str) -> str:
    if pd.isna(value):
        return "NA"
    no_decimal_indicators = {"Hot_D", "Huglin_Index", "Climatic_Dryness_Index", "stress_climatique"}
    if indicator in no_decimal_indicators:
        return f"{value:.0f}"
    return f"{value:.1f}"


def ensure_hist_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "cluster" in df.columns:
        df["cluster"] = safe_numeric(df["cluster"]).astype("Int64")
    if "Year" in df.columns:
        df["Year"] = safe_numeric(df["Year"]).astype("Int64")
    if "Municipality" in df.columns:
        df["Municipality"] = df["Municipality"].astype(str)
    
    numeric_cols = [
        "code_departement", "latitude", "longitude", "temp_moyenne", "tmax_mean", "tmin_mean",
        "precipitation_total", "precipitation_total_avril_septembre", "amplitude_thermique",
        "Huglin_Index", "Hot_D", "Very_Hot_D", "Frost_D", "Late_Frost", "stress_climatique",
        "deficit_hydrique", "Climatic_Dryness_Index", "Soil_Water_Stock", "Soil_pH", "jours_secs", "jours_pluie",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    return df


def ensure_proj_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "zone" in df.columns:
        df["zone"] = safe_numeric(df["zone"]).astype("Int64")
    if "year" in df.columns:
        df["year"] = safe_numeric(df["year"]).astype("Int64")
    if "commune_code" in df.columns:
        df["commune_code"] = df["commune_code"].astype(str)
    if "code_departement" in df.columns:
        df["code_departement"] = safe_numeric(df["code_departement"]).astype("Int64")
    for col in ["latitude", "longitude", "temp_moyenne", "tmax_mean", "tmin_mean", "precipitation_total"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    return df


@st.cache_data
def load_climat() -> pd.DataFrame:
    """Charge les données climatiques historiques en utilisant la connexion partagée"""
    try:
        conn = get_conn()
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if "climat_final" in table_names:
            df = conn.execute("SELECT * FROM climat_final").df()
        elif "climat_rendement_geo" in table_names:
            df = conn.execute("SELECT * FROM climat_rendement_geo").df()
        else:
            st.error("Table climat_final ou climat_rendement_geo non trouvee")
            return pd.DataFrame()
        
        if df.empty:
            st.warning("Aucune donnee climatique disponible")
            return df
        
        return ensure_hist_schema(df)
    except Exception as e:
        st.error(f"Erreur chargement donnees climatiques: {e}")
        return pd.DataFrame()


@st.cache_data
def load_projection() -> pd.DataFrame:
    """Charge les données de projection climatique en utilisant la connexion partagée"""
    try:
        conn = get_conn()
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if "climat_projection_final" in table_names:
            df = conn.execute("SELECT * FROM climat_projection_final").df()
        else:
            return pd.DataFrame()
        
        if df.empty:
            return df
        
        return ensure_proj_schema(df)
    except Exception as e:
        st.warning(f"Donnees de projection non disponibles: {e}")
        return pd.DataFrame()


@st.cache_data
def load_geo_zone() -> pd.DataFrame:
    """Charge les données géographiques des zones en utilisant la connexion partagée"""
    try:
        conn = get_conn()
        tables = conn.execute("SHOW TABLES").fetchall()
        table_names = [t[0] for t in tables]
        
        if "geo_zone" in table_names:
            df = conn.execute("SELECT * FROM geo_zone").df()
        else:
            st.error("Table geo_zone non trouvee")
            return pd.DataFrame()
        
        if "zone" in df.columns:
            df["zone"] = safe_numeric(df["zone"]).astype("Int64")
        if "commune_code" in df.columns:
            df["commune_code"] = df["commune_code"].astype(str)
        return df
    except Exception as e:
        st.error(f"Erreur chargement donnees geographiques: {e}")
        return pd.DataFrame()


@st.cache_data
def load_communes_geojson() -> gpd.GeoDataFrame:
    """Charge le GeoJSON des communes"""
    try:
        url = "https://raw.githubusercontent.com/Juralexx/france-geojson-datas/master/communes.geojson"
        gdf = gpd.read_file(url)
        gdf["commune_code"] = gdf["code"].astype(str)
        gdf = gdf[gdf["commune_code"].str[:2].isin(["11", "30", "34", "66"])].copy()
        return gdf
    except Exception as e:
        st.error(f"Erreur chargement GeoJSON: {e}")
        return gpd.GeoDataFrame()


# Chargement des donnees
with st.spinner("Chargement des donnees climatiques..."):
    df_climat = load_climat()
    df_proj = load_projection()
    df_geo_zone = load_geo_zone()

if df_climat.empty:
    st.error("Impossible de charger les donnees climatiques. Veuillez verifier la base de donnees.")
    st.stop()


# =====================================================
# FONCTIONS DE VISUALISATION
# =====================================================

def create_zone_map(df_zone_values: pd.DataFrame, indicator: str, title_label: str):
    """Crée une carte Folium des zones avec tooltips"""
    communes_gdf = load_communes_geojson()
    if communes_gdf.empty:
        return folium.Map(location=[43.7, 3.5], zoom_start=8)
    
    geo_zone = df_geo_zone[["commune_code", "zone"]].drop_duplicates().copy()
    gdf = communes_gdf.merge(geo_zone, on="commune_code", how="left")
    gdf["zone"] = safe_numeric(gdf["zone"]).astype("Int64")
    gdf = gdf.dropna(subset=["zone"]).copy()
    
    if gdf.empty:
        return folium.Map(location=[43.7, 3.5], zoom_start=8)
    
    zone_geom = gdf[["zone", "geometry"]].dissolve(by="zone", as_index=False)
    values_df = df_zone_values[["zone", indicator]].copy()
    values_df["zone"] = safe_numeric(values_df["zone"]).astype("Int64")
    values_df[indicator] = safe_numeric(values_df[indicator])
    
    zones_gdf = zone_geom.merge(values_df, on="zone", how="left")
    zones_gdf["zone_str"] = zones_gdf["zone"].astype("Int64").astype(str)
    zones_gdf["zone_label"] = zones_gdf["zone_str"].map(ZONE_LABELS)
    zones_gdf["indicator_fmt"] = zones_gdf[indicator].apply(lambda x: format_value(x, indicator))
    
    bounds = zones_gdf.total_bounds
    center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    m = folium.Map(location=center, zoom_start=8, tiles="CartoDB positron")
    
    min_val, max_val = float(zones_gdf[indicator].min()), float(zones_gdf[indicator].max())
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        max_val = min_val + 1.0 if pd.notna(min_val) else 1.0
        min_val = min_val if pd.notna(min_val) else 0.0
    
    colormap = linear.YlOrRd_09.scale(min_val, max_val)
    colormap.caption = indicator_label(indicator)
    
    def style_function(feature):
        val = feature["properties"].get(indicator)
        zone_str = str(feature["properties"].get("zone"))
        border_color = ZONE_COLOR_MAP.get(zone_str, "#BDBDBD")
        return {
            "fillColor": "#D9D9D9" if val is None or pd.isna(val) else colormap(val),
            "color": border_color,
            "weight": 3,
            "fillOpacity": 0.8
        }
    
    folium.GeoJson(
        zones_gdf,
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone_label", "indicator_fmt"],
            aliases=["Zone", indicator_label(indicator)],
            localize=False,
            sticky=True,
            style="""
                background-color: white;
                border: 2px solid #333;
                border-radius: 5px;
                font-size: 12px;
                padding: 5px;
            """
        ),
        highlight_function=lambda x: {'weight': 3, 'color': 'black', 'fillOpacity': 0.9}
    ).add_to(m)
    colormap.add_to(m)
    
    return m


def plot_temperature_trend_with_smoothing(df: pd.DataFrame, selected_zones: list[int]):
    """Graphique des tendances de température avec lissage Savitzky-Golay et tooltips"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for zone in selected_zones:
        df_zone = df[df["cluster"] == zone].copy()
        if df_zone.empty:
            continue
        
        if "Year" not in df_zone.columns or "temp_moyenne" not in df_zone.columns:
            continue
        
        grouped = df_zone.groupby("Year")["temp_moyenne"].mean().reset_index().sort_values("Year")
        grouped["Year"] = grouped["Year"].astype(int)
        
        if len(grouped) < 5:
            continue
        
        years = grouped["Year"].values
        temps = grouped["temp_moyenne"].values
        
        # Lissage Savitzky-Golay
        window = min(5, len(temps) if len(temps) % 2 == 1 else len(temps) - 1)
        if window >= 3 and window % 2 == 1:
            smoothed = savgol_filter(temps, window, 2)
        else:
            smoothed = temps
        
        zone_str = str(int(zone))
        color = ZONE_COLOR_MAP.get(zone_str, "#333333")
        
        # Données avec tooltip
        ax.plot(years, temps, 'o', color=color, alpha=0.3, markersize=4, label="_nolegend_")
        line, = ax.plot(years, smoothed, '-', linewidth=2.5, color=color, 
                        label=ZONE_LABELS.get(zone_str, f"Zone {zone_str}"),
                        picker=True, pickradius=5)
        
        # Tendance linéaire
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, temps)
        ax.plot(years, slope * years + intercept, '--', linewidth=1, color=color, alpha=0.5)
        
        # Affichage de la pente
        ax.text(years[-1], temps[-1], f"  +{slope:.3f}°C/an", fontsize=8, color=color, alpha=0.7)
    
    ax.set_title("Tendance des temperatures avec lissage (Savitzky-Golay)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annee", fontsize=12)
    ax.set_ylabel("Temperature moyenne (°C)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def plot_radar_chart(df: pd.DataFrame, selected_zones: list[int]):
    """Diagramme en radar pour comparer les zones avec tooltips"""
    indicators = ["temp_moyenne", "Huglin_Index", "Very_Hot_D", "deficit_hydrique", "jours_pluie"]
    available_indicators = [i for i in indicators if i in df.columns]
    
    if not available_indicators:
        return None
    
    zone_profiles = {}
    for zone in selected_zones:
        df_zone = df[df["cluster"] == zone].copy()
        if not df_zone.empty:
            values = []
            for ind in available_indicators:
                val = df_zone[ind].mean()
                if pd.notna(val):
                    values.append(val)
                else:
                    values.append(0)
            zone_profiles[zone] = values
    
    if not zone_profiles:
        return None
    
    # Normalisation
    all_values = np.array(list(zone_profiles.values()))
    if all_values.size > 0:
        min_vals = all_values.min(axis=0)
        max_vals = all_values.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1
        
        for zone in zone_profiles:
            zone_profiles[zone] = (zone_profiles[zone] - min_vals) / ranges
    
    angles = np.linspace(0, 2 * np.pi, len(available_indicators), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    for zone, values in zone_profiles.items():
        values_plot = list(values) + [values[0]]
        zone_str = str(int(zone))
        color = ZONE_COLOR_MAP.get(zone_str, "#333333")
        label = ZONE_LABELS.get(zone_str, f"Zone {zone_str}")
        ax.plot(angles, values_plot, 'o-', linewidth=2, color=color, label=label)
        ax.fill(angles, values_plot, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([indicator_label(i) for i in available_indicators], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Comparaison des zones - Profil climatique normalise", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    return fig


def plot_boxplots_by_zone(df: pd.DataFrame, selected_zones: list[int], indicator: str):
    """Boxplots des indicateurs par zone avec tooltips"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    data_by_zone = []
    labels = []
    colors = []
    
    for zone in selected_zones:
        df_zone = df[df["cluster"] == zone].copy()
        if not df_zone.empty and indicator in df_zone.columns:
            values = df_zone[indicator].dropna()
            if not values.empty:
                data_by_zone.append(values)
                zone_str = str(int(zone))
                labels.append(ZONE_LABELS.get(zone_str, f"Zone {zone_str}"))
                colors.append(ZONE_COLOR_MAP.get(zone_str, "#333333"))
    
    if data_by_zone:
        # Correction du warning matplotlib
        bp = ax.boxplot(data_by_zone, patch_artist=True, tick_labels=labels)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Ajout de la médiane sur chaque boîte
        for i, (box, median) in enumerate(zip(bp['boxes'], bp['medians'])):
            median.set_color('black')
            median.set_linewidth(2)
        
        ax.set_title(f"Distribution de {indicator_label(indicator)} par zone", fontsize=14, fontweight="bold")
        ax.set_ylabel(indicator_label(indicator), fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
    
    fig.tight_layout()
    return fig


def plot_correlation_matrix(df: pd.DataFrame, selected_zones: list[int]):
    """Matrice de corrélation des indicateurs climatiques"""
    indicators = ["temp_moyenne", "tmax_mean", "tmin_mean", "precipitation_total", 
                  "Huglin_Index", "Very_Hot_D", "deficit_hydrique", "jours_pluie"]
    available_indicators = [i for i in indicators if i in df.columns]
    
    if len(available_indicators) < 2:
        return None
    
    df_filtered = df[df["cluster"].isin(selected_zones)][available_indicators].dropna()
    
    if df_filtered.empty:
        return None
    
    corr_matrix = df_filtered.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={'size': 9}, ax=ax)
    
    ax.set_title("Matrice de correlation des indicateurs climatiques", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_anomaly_detection(df: pd.DataFrame, selected_zones: list[int], indicator: str):
    """Détection des anomalies climatiques avec tooltips"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for zone in selected_zones:
        df_zone = df[df["cluster"] == zone].copy()
        if df_zone.empty or indicator not in df_zone.columns:
            continue
        
        if "Year" not in df_zone.columns:
            continue
        
        grouped = df_zone.groupby("Year")[indicator].mean().reset_index().sort_values("Year")
        grouped["Year"] = grouped["Year"].astype(int)
        
        if len(grouped) < 3:
            continue
        
        mean_val = grouped[indicator].mean()
        std_val = grouped[indicator].std()
        
        years = grouped["Year"].values
        values = grouped[indicator].values
        
        zone_str = str(int(zone))
        color = ZONE_COLOR_MAP.get(zone_str, "#333333")
        label = ZONE_LABELS.get(zone_str, f"Zone {zone_str}")
        
        # Tracé principal
        line, = ax.plot(years, values, 'o-', linewidth=2, color=color, label=label, picker=True)
        
        # Moyenne
        ax.axhline(mean_val, linestyle='--', color=color, alpha=0.5)
        
        # Bornes à 2 écarts-types
        ax.axhline(mean_val + 2*std_val, linestyle=':', color=color, alpha=0.3)
        ax.axhline(mean_val - 2*std_val, linestyle=':', color=color, alpha=0.3)
        
        # Marquage des anomalies
        anomalies = np.where(np.abs(values - mean_val) > 2*std_val)[0]
        if len(anomalies) > 0:
            ax.scatter(years[anomalies], values[anomalies], s=100, c='red', marker='o', zorder=5, 
                      label='Anomalie' if zone == selected_zones[0] else "")
    
    ax.set_title(f"Detection d'anomalies - {indicator_label(indicator)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annee", fontsize=12)
    ax.set_ylabel(indicator_label(indicator), fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def plot_moving_average(df: pd.DataFrame, selected_zones: list[int], indicator: str, window: int = 5):
    """Moyenne mobile pour lisser les tendances avec tooltips"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for zone in selected_zones:
        df_zone = df[df["cluster"] == zone].copy()
        if df_zone.empty or indicator not in df_zone.columns:
            continue
        
        if "Year" not in df_zone.columns:
            continue
        
        grouped = df_zone.groupby("Year")[indicator].mean().reset_index().sort_values("Year")
        grouped["Year"] = grouped["Year"].astype(int)
        
        if len(grouped) < window:
            continue
        
        years = grouped["Year"].values
        values = grouped[indicator].values
        
        # Moyenne mobile
        ma = np.convolve(values, np.ones(window)/window, mode='valid')
        ma_years = years[window-1:]
        
        zone_str = str(int(zone))
        color = ZONE_COLOR_MAP.get(zone_str, "#333333")
        label = ZONE_LABELS.get(zone_str, f"Zone {zone_str}")
        
        ax.plot(years, values, 'o', color=color, alpha=0.3, markersize=4, label="_nolegend_")
        ax.plot(ma_years, ma, '-', linewidth=2.5, color=color, label=f"{label} (MA{window})")
    
    ax.set_title(f"Moyenne mobile ({window} ans) - {indicator_label(indicator)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annee", fontsize=12)
    ax.set_ylabel(indicator_label(indicator), fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def plot_historical_curves(df: pd.DataFrame, selected_zones: list[int], indicator: str):
    """Courbes historiques avec tooltips"""
    fig, ax = plt.subplots(figsize=(11, 5))
    
    for zone in selected_zones:
        df_zone = df[df["cluster"] == zone].copy()
        if df_zone.empty or indicator not in df_zone.columns:
            continue
        
        if "Year" not in df_zone.columns:
            continue
        
        grouped = df_zone.groupby("Year")[indicator].mean().reset_index().dropna().sort_values("Year")
        if grouped.empty:
            continue
        
        grouped["Year"] = grouped["Year"].astype(int)
        zone_str = str(int(zone))
        label = ZONE_LABELS.get(zone_str, f"Zone {zone_str}")
        
        ax.plot(
            grouped["Year"],
            grouped[indicator],
            marker="o",
            linewidth=2,
            color=ZONE_COLOR_MAP.get(zone_str, "#333333"),
            label=label,
            picker=True,
            pickradius=5
        )
    
    ax.set_title(f"Historique - {indicator_label(indicator)}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Annee", fontsize=12)
    ax.set_ylabel(indicator_label(indicator), fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


def plot_scenario_comparison(df_proj: pd.DataFrame, selected_zones: list[int], indicator: str, period: str):
    """Graphique de comparaison des scénarios climatiques avec tooltips"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scenarios = ["optimiste", "neutre", "pessimiste"]
    sorted_zones = sorted(selected_zones)
    x = np.arange(len(sorted_zones))
    width = 0.25
    
    for i, scenario in enumerate(scenarios):
        vals = []
        for zone in sorted_zones:
            subset = df_proj[(df_proj["zone"] == zone) & (df_proj["scenario"] == scenario) & (df_proj["periode"] == period)]
            if not subset.empty and indicator in subset.columns:
                vals.append(subset[indicator].mean())
            else:
                vals.append(np.nan)
        
        bars = ax.bar(x + (i - 1) * width, vals, width=width, 
                      label=scenario.capitalize(), color=SCENARIO_COLOR_MAP[scenario], alpha=0.85)
        
        # Ajout des valeurs sur les barres
        for bar, val in zip(bars, vals):
            if not pd.isna(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"Zone {int(z)}" for z in sorted_zones])
    ax.set_title(f"Comparaison des scenarios - {indicator_label(indicator)} - {period}", fontsize=14, fontweight="bold")
    ax.set_ylabel(indicator_label(indicator), fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:
    st.header("Filtres")
    st.divider()
    
    if "cluster" in df_climat.columns:
        available_zones = sorted([int(z) for z in df_climat["cluster"].dropna().unique()])
    else:
        available_zones = []
    
    if available_zones:
        selected_zones = st.multiselect(
            "Zones pedoclimatiques",
            options=available_zones,
            default=available_zones[:3] if len(available_zones) >= 3 else available_zones
        )
    else:
        selected_zones = []
        st.error("Aucune zone disponible")
    
    st.divider()
    
    if "Year" in df_climat.columns:
        available_years = sorted([int(y) for y in df_climat["Year"].dropna().unique()])
        if available_years:
            selected_year = st.selectbox("Annee historique", options=available_years, index=len(available_years) - 1)
        else:
            selected_year = None
    else:
        selected_year = None
    
    use_last_5y_mean = st.checkbox("Moyenne des 5 dernieres annees", value=False)
    
    st.divider()
    
    available_indicators = [i for i in HISTORICAL_INDICATORS if i in df_climat.columns]
    if available_indicators:
        selected_indicator = st.selectbox(
            "Indicateur historique",
            options=available_indicators,
            format_func=indicator_label
        )
    else:
        selected_indicator = None

if not selected_zones:
    st.warning("Selectionnez au moins une zone pour commencer l'analyse.")
    st.stop()


# =====================================================
# ANALYSE HISTORIQUE
# =====================================================

st.header("Analyse historique")
st.markdown("---")

if use_last_5y_mean and "Year" in df_climat.columns:
    year_max = int(df_climat["Year"].dropna().max())
    year_min = year_max - 4
    hist_filtered = df_climat[(df_climat["cluster"].isin(selected_zones)) & (df_climat["Year"].between(year_min, year_max))].copy()
    period_label = f"Moyenne {year_min}-{year_max}"
elif selected_year and "Year" in df_climat.columns:
    hist_filtered = df_climat[(df_climat["cluster"].isin(selected_zones)) & (df_climat["Year"] == selected_year)].copy()
    period_label = str(int(selected_year))
else:
    hist_filtered = df_climat[df_climat["cluster"].isin(selected_zones)].copy()
    period_label = "Toutes annees"

col1, col2 = st.columns(2)
with col1:
    if selected_indicator and selected_indicator in hist_filtered.columns:
        metric_val = hist_filtered[selected_indicator].mean()
        st.metric(label=f"{indicator_label(selected_indicator)} - {period_label}", value=format_value(metric_val, selected_indicator))
    else:
        st.metric(label="Indicateur", value="N/A")

# Tableau
st.subheader("Tableau par zone")
if selected_indicator and selected_indicator in df_climat.columns:
    if use_last_5y_mean and "Year" in df_climat.columns:
        table_zone = df_climat[df_climat["Year"].between(year_min, year_max)].groupby("cluster", as_index=False)[selected_indicator].mean().rename(columns={"cluster": "zone"}).sort_values("zone")
    elif selected_year and "Year" in df_climat.columns:
        table_zone = df_climat[df_climat["Year"] == selected_year].groupby("cluster", as_index=False)[selected_indicator].mean().rename(columns={"cluster": "zone"}).sort_values("zone")
    else:
        table_zone = df_climat.groupby("cluster", as_index=False)[selected_indicator].mean().rename(columns={"cluster": "zone"}).sort_values("zone")
    
    if not table_zone.empty:
        table_zone_display = table_zone.copy()
        table_zone_display["zone"] = table_zone_display["zone"].apply(lambda z: f"Zone {int(z)}")
        table_zone_display[selected_indicator] = table_zone_display[selected_indicator].apply(lambda x: format_value(x, selected_indicator))
        st.dataframe(table_zone_display, width="stretch", hide_index=True)

# Carte historique
st.subheader("Carte historique par zone")
map_indicator_hist = st.selectbox("Indicateur cartographie (historique)", options=MAP_INDICATORS, format_func=indicator_label, key="map_indicator_hist")

try:
    if use_last_5y_mean and "Year" in df_climat.columns:
        zone_values_hist = df_climat[df_climat["Year"].between(year_min, year_max)].groupby("cluster", as_index=False)[map_indicator_hist].mean().rename(columns={"cluster": "zone"})
        map_title = f"Carte historique - {indicator_label(map_indicator_hist)} - moyenne {year_min}-{year_max}"
    elif selected_year and "Year" in df_climat.columns:
        zone_values_hist = df_climat[df_climat["Year"] == selected_year].groupby("cluster", as_index=False)[map_indicator_hist].mean().rename(columns={"cluster": "zone"})
        map_title = f"Carte historique - {indicator_label(map_indicator_hist)} - {int(selected_year)}"
    else:
        zone_values_hist = df_climat.groupby("cluster", as_index=False)[map_indicator_hist].mean().rename(columns={"cluster": "zone"})
        map_title = f"Carte historique - {indicator_label(map_indicator_hist)} - moyenne toutes annees"
    
    map_hist = create_zone_map(zone_values_hist, map_indicator_hist, map_title)
    components.html(map_hist._repr_html_(), height=650)
except Exception as e:
    st.error(f"Erreur carte : {e}")

# Courbes historiques
st.subheader("Courbes historiques")
if selected_indicator and "Year" in df_climat.columns:
    try:
        fig_hist = plot_historical_curves(df_climat, selected_zones, selected_indicator)
        st.pyplot(fig_hist)
        plt.close(fig_hist)
    except Exception as e:
        st.error(f"Erreur graphique historique : {e}")


# =====================================================
# NOUVEAUX GRAPHIQUES - ANALYSES APPROFONDIES
# =====================================================

st.header("Analyses climatiques approfondies")
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Tendances", "Radar", "Boxplots", "Correlations", "Anomalies"
])

with tab1:
    st.subheader("Tendances des temperatures avec lissage")
    if "temp_moyenne" in df_climat.columns and "Year" in df_climat.columns:
        try:
            fig_trend = plot_temperature_trend_with_smoothing(df_climat, selected_zones)
            st.pyplot(fig_trend)
            plt.close(fig_trend)
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.info("Donnees de temperature non disponibles")
    
    st.subheader("Moyenne mobile")
    if selected_indicator and "Year" in df_climat.columns:
        ma_window = st.slider("Fenetre de moyenne mobile (annees)", 3, 10, 5, key="ma_window")
        try:
            fig_ma = plot_moving_average(df_climat, selected_zones, selected_indicator, ma_window)
            st.pyplot(fig_ma)
            plt.close(fig_ma)
        except Exception as e:
            st.error(f"Erreur : {e}")

with tab2:
    st.subheader("Comparaison des zones - Diagramme en radar")
    try:
        fig_radar = plot_radar_chart(df_climat, selected_zones)
        if fig_radar:
            st.pyplot(fig_radar)
            plt.close(fig_radar)
        else:
            st.info("Donnees insuffisantes pour le diagramme en radar")
    except Exception as e:
        st.error(f"Erreur : {e}")

with tab3:
    st.subheader("Distribution des indicateurs par zone")
    if selected_indicator:
        try:
            fig_boxplot = plot_boxplots_by_zone(df_climat, selected_zones, selected_indicator)
            st.pyplot(fig_boxplot)
            plt.close(fig_boxplot)
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.info("Selectionnez un indicateur")

with tab4:
    st.subheader("Matrice de correlation des indicateurs")
    try:
        fig_corr = plot_correlation_matrix(df_climat, selected_zones)
        if fig_corr:
            st.pyplot(fig_corr)
            plt.close(fig_corr)
        else:
            st.info("Donnees insuffisantes pour la matrice de correlation")
    except Exception as e:
        st.error(f"Erreur : {e}")

with tab5:
    st.subheader("Detection d'anomalies climatiques")
    if selected_indicator and "Year" in df_climat.columns:
        try:
            fig_anomaly = plot_anomaly_detection(df_climat, selected_zones, selected_indicator)
            st.pyplot(fig_anomaly)
            plt.close(fig_anomaly)
        except Exception as e:
            st.error(f"Erreur : {e}")
    else:
        st.info("Selectionnez un indicateur avec des donnees annuelles")


# =====================================================
# SCENARIOS CLIMATIQUES
# =====================================================

if not df_proj.empty:
    st.header("Comparaison des scenarios climatiques")
    st.markdown("---")
    
    if "periode" in df_proj.columns:
        scenario_period = st.radio("Periode de projection", options=["2021-2040", "2041-2060"], horizontal=True)
        
        st.subheader("Graphique comparatif")
        compare_indicator = st.selectbox("Indicateur a comparer", options=SCENARIO_INDICATORS, format_func=indicator_label, key="scenario_indicator")
        
        try:
            fig_scenario = plot_scenario_comparison(df_proj, selected_zones, compare_indicator, scenario_period)
            st.pyplot(fig_scenario)
            plt.close(fig_scenario)
        except Exception as e:
            st.error(f"Erreur graphique scenario : {e}")
        
        st.subheader("Tableau comparatif par zone")
        proj_period = df_proj[df_proj["periode"] == scenario_period].copy()
        
        if not proj_period.empty:
            scenario_table = []
            for zone in selected_zones:
                for scenario in ["optimiste", "neutre", "pessimiste"]:
                    subset = proj_period[(proj_period["zone"] == zone) & (proj_period["scenario"] == scenario)].copy()
                    if not subset.empty:
                        row = {"zone": int(zone), "scenario": scenario}
                        for col in SCENARIO_INDICATORS:
                            if col in subset.columns:
                                row[col] = subset[col].mean()
                            else:
                                row[col] = np.nan
                        scenario_table.append(row)
            
            if scenario_table:
                scenario_df = pd.DataFrame(scenario_table)
                display_df = scenario_df.copy()
                display_df["zone"] = display_df["zone"].apply(lambda z: f"Zone {int(z)}")
                for col in SCENARIO_INDICATORS:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: format_value(x, col))
                st.dataframe(display_df, width="stretch", hide_index=True)
            
            # Carte projetee
            st.subheader("Carte projetee par zone")
            map_scenario = st.selectbox("Scenario cartographie", options=["optimiste", "neutre", "pessimiste"], key="map_scenario")
            map_indicator_proj = st.selectbox("Indicateur cartographie (projection)", options=MAP_INDICATORS, format_func=indicator_label, key="map_indicator_proj")
            
            try:
                values_proj_zone = proj_period[proj_period["scenario"] == map_scenario].groupby("zone", as_index=False)[map_indicator_proj].mean()
                if not values_proj_zone.empty:
                    map_proj = create_zone_map(
                        values_proj_zone,
                        map_indicator_proj,
                        f"Carte projetee - {indicator_label(map_indicator_proj)} - {map_scenario} - {scenario_period}",
                    )
                    components.html(map_proj._repr_html_(), height=650)
            except Exception as e:
                st.error(f"Erreur carte projetee : {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>Modelisation climatique - Pays d'Oc | Donnees mises a jour regulierement</p>
    <p style="font-size: 0.75rem;">© 2024 - Analyse climatique et projections</p>
</div>
""", unsafe_allow_html=True)

gc.collect()