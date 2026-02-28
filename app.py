import streamlit as st

st.set_page_config(page_title="Life Insurance — Actuarial & ML", layout="wide")

st.title("Life Insurance — Actuarial Engine vs ML Engine")
st.write(
    """
Application Streamlit avec :
- **Moteur Actuariat** (pricing + réserves) via table TGF05 générationnelle
- **Moteur ML** (prédiction des mêmes quantités)
- **Performance & Training** (dataset, entraînement, métriques, learning curves, feature importance)
"""
)
