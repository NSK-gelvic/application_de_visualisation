# modules/prediction.py
"""
Modèles de prédiction - Version simplifiée avec Random Forest uniquement
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Optional

from modules.utils import zone_sort_key


def run_prediction(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Prédiction avec Random Forest uniquement
    """
    st.subheader("Paramètres de prédiction")

    # Paramètres
    col1, col2, col3 = st.columns(3)
    
    with col1:
        var_lbl = st.radio("Variable à prédire", ["Rendement", "Volume"], horizontal=True, key="pred_var")
    
    with col2:
        group = st.radio(
            "Regrouper par", 
            ["Couleur", "Département", "Zone", "Cépage"], 
            horizontal=True, 
            key="pred_group"
        )
    
    with col3:
        horizon = st.slider("Horizon de prédiction (années)", 1, 5, 3, key="pred_h")

    # Mapping des colonnes
    col_map = {
        "Couleur": "code_couleur",
        "Département": "code_departement",
        "Zone": "Zone",
        "Cépage": "code_cepage"
    }[group]

    var = "rendement" if var_lbl == "Rendement" else "volume"
    mvt = "DECR" if var == "rendement" else "REVE"

    base = df[df["type_mvt"] == mvt].dropna(subset=[var, col_map]).copy()

    if base.empty:
        st.warning("Pas assez de données pour la prédiction")
        return None

    # Sélection
    unique_values = sorted(
        base[col_map].dropna().unique(),
        key=(zone_sort_key if col_map == "Zone" else None)
    )
    
    sel = st.multiselect(
        "Éléments à analyser",
        unique_values,
        default=unique_values[:min(2, len(unique_values))],
        key="pred_sel"
    )

    if not sel:
        return None

    # Graphique
    fig = go.Figure()
    fig.update_layout(template="plotly_white")
    
    results = []
    test_size = 3

    for g in sel:
        dfg = base[base[col_map] == g].groupby("annee")[var].mean().reset_index().sort_values("annee")
        
        if len(dfg) < 6:
            st.warning(f"Pas assez de données pour {g} (minimum 6 années requises)")
            continue

        # Série complète
        yrs = np.arange(dfg["annee"].min(), dfg["annee"].max() + 1)
        dfg_full = dfg.set_index("annee").reindex(yrs).reset_index().rename(columns={"index": "annee"})
        dfg_full[col_map] = g

        # Données historiques
        fig.add_trace(go.Scatter(
            x=dfg_full["annee"],
            y=dfg_full[var],
            mode="lines+markers",
            name=f"{g} - Historique",
            line=dict(width=2)
        ))

        # Train/test
        train = dfg.iloc[:-test_size]
        test = dfg.iloc[-test_size:]
        y_true = test[var].values

        if len(train) < 3 or len(test) == 0:
            continue

        # Random Forest
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            X_train = train[["annee"]].values
            X_test = test[["annee"]].values
            rf.fit(X_train, train[var].values)
            y_rf = rf.predict(X_test)
            
            # Prédictions futures
            last_year = dfg["annee"].max()
            future_years = np.arange(last_year + 1, last_year + horizon + 1)
            X_future = future_years.reshape(-1, 1)
            y_future = rf.predict(X_future)
            
            # Test (points)
            fig.add_trace(go.Scatter(
                x=test["annee"],
                y=y_rf,
                mode="markers",
                name=f"{g} - Test",
                marker=dict(size=8, symbol='x')
            ))
            
            # Prédictions futures
            fig.add_trace(go.Scatter(
                x=future_years,
                y=y_future,
                mode="lines+markers",
                name=f"{g} - Prévision",
                line=dict(dash="dash", width=2)
            ))
            
            results.append((
                g, "Random Forest",
                mean_absolute_error(y_true, y_rf),
                np.sqrt(mean_squared_error(y_true, y_rf)),
                r2_score(y_true, y_rf)
            ))
        except Exception as e:
            st.error(f"Erreur pour {g}: {e}")

    # Mise en page
    fig.update_layout(
        title=f"Prédiction {var_lbl} avec Random Forest",
        xaxis_title="Année",
        yaxis_title=f"{var_lbl} ({'hl/ha' if var == 'rendement' else 'hl'})",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Résultats
    if results:
        res = pd.DataFrame(results, columns=["Groupe", "Modèle", "MAE", "RMSE", "R2"]).round(3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance du modèle")
            st.dataframe(res, use_container_width=True)
        
        with col2:
            st.subheader("Qualité de prédiction")
            avg_rmse = res['RMSE'].mean()
            avg_r2 = res['R2'].mean()
            st.metric("RMSE moyen", f"{avg_rmse:.2f}")
            st.metric("R² moyen", f"{avg_r2:.2f}")
            
            # Meilleure performance
            best_idx = res['RMSE'].idxmin()
            st.info(f"Meilleure prédiction : {res.loc[best_idx, 'Groupe']} (RMSE={res.loc[best_idx, 'RMSE']:.2f})")
        
        return res
    
    return None