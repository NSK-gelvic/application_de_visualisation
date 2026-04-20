# -*- coding: utf-8 -*-

import streamlit as st

st.title("Accueil")

st.markdown(
    """
    ## Présentation du projet

    Cet observatoire permet d'analyser les relations entre :
    - le **climat**
    - le **rendement viticole**
    - les **zones pédoclimatiques**
    - les **départements du Pays d'Oc**

    ## Les pages disponibles

    ### 1. Rendement
    Analyse des volumes, surfaces, rendements, zones, départements, couleurs.

    ### 2. Climat
    Analyse des indicateurs climatiques historiques et projetés.

    ### 3. Analyse climat-rendement
    Étude des liens entre variables climatiques et rendement par année, zone et département.

    ## Objectif
    Mieux comprendre l'impact du changement climatique sur la viticulture du Pays d'Oc.
    """
)
