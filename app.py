# -*- coding: utf-8 -*-

import streamlit as st

st.set_page_config(
    page_title="Observatoire Viticole - Pays d'Oc",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .block-container {
            padding-top: 1.2rem;
        }
        .stMetric {
            background-color: white;
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍇 Observatoire Viticole - Pays d'Oc")
st.markdown("Analyse du climat, du rendement et de leurs interactions.")
st.info("Utilise le menu de navigation de Streamlit pour accéder aux 4 pages.")