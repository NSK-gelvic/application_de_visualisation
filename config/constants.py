# -*- coding: utf-8 -*-

from pathlib import Path

# =====================================================
# RACINE PROJET
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# =====================================================
# FICHIERS PRINCIPAUX
# =====================================================

DB_PATH = BASE_DIR / "db" / "mvttdb.duckdb"
PARQUET_PATH = BASE_DIR / "partie_rendement_et_volume" / "dataset_mvttdb_final.parquet"

DB_FILE = str(DB_PATH)
PARQUET_FILE = str(PARQUET_PATH)

# =====================================================
# DOSSIER GEOJSON
# =====================================================

GEOJSON_DIR = BASE_DIR / "partie_rendement_et_volume" / "assets" / "geojson"

GEOJSON_FILES = {
    "departements": GEOJSON_DIR / "departements_languedoc.geojson",
    "zones": GEOJSON_DIR / "zones_pedoclimatiques.geojson",
    "communes": GEOJSON_DIR / "communes_languedoc.geojson",
}

# =====================================================
# CONSTANTES METIER
# =====================================================

# Couleurs des vins
COLOR_MAP = {
    "BL": "#F4D03F",  # Blanc - jaune
    "RG": "#A93226",  # Rouge - rouge
    "RS": "#F1948A",  # Rose - rose
}

# Couleurs des zones pedoclimatiques (1 a 7)
ZONE_COLOR_MAP = {
    "1": "#000000",  # Zone 1 - noir
    "2": "#FF0000",  # Zone 2 - rouge
    "3": "#1A8F2A",  # Zone 3 - vert
    "4": "#0033CC",  # Zone 4 - bleu
    "5": "#AFC6D9",  # Zone 5 - bleu clair
    "6": "#7A1FA2",  # Zone 6 - violet
    "7": "#FFD800",  # Zone 7 - jaune
}

# Labels des zones pedoclimatiques (1 a 7)
ZONE_LABELS = {
    "1": "Zone 1 : zone humide de l'arriere-pays",
    "2": "Zone 2 : montagne, sols acides et peu profonds",
    "3": "Zone 3 : Piedmont, reserve utile limitee",
    "4": "Zone 4 : froide et seche autour du Pic Saint-Loup",
    "5": "Zone 5 : sols de qualite moyenne dans l'arriere-pays",
    "6": "Zone 6 : sols profonds, cotes temperees",
    "7": "Zone 7 : tres chaud, sols profonds",
}

# Liste des departements
DEPARTEMENTS = ["11", "30", "34", "66"]

# Couleurs des departements
DEPT_COLOR_MAP = {
    "11": "#3498db",  # Aude - bleu
    "30": "#2ecc71",  # Gard - vert
    "34": "#e74c3c",  # Herault - rouge
    "66": "#f39c12",  # Pyrenees-Orientales - orange
}

# Noms des departements
DEPT_NAMES = {
    "11": "Aude",
    "30": "Gard",
    "34": "Herault",
    "66": "Pyrenees-Orientales"
}

# =====================================================
# CHECK
# =====================================================

def check_paths():
    missing = []

    if not DB_PATH.exists():
        missing.append(f"DB manquante: {DB_PATH}")

    if not PARQUET_PATH.exists():
        missing.append(f"Parquet manquant: {PARQUET_PATH}")

    for name, path in GEOJSON_FILES.items():
        if not Path(path).exists():
            missing.append(f"GEOJSON manquant ({name}): {path}")

    return missing