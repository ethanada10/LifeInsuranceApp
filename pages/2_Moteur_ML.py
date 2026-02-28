import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core.actuarial import (
    load_tgf05_lx,
    build_qx_from_lx,
    single_premium,
    annual_premium,
    reserve_Vt,
)
from core.ml import load_models, predict_all

MODELS_DIR = "models"


def find_table_path(filename: str = "TGF05-TGH05.xls") -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        Path.cwd() / filename,
        Path.cwd() / "data" / filename,
        here / filename,
        here / "data" / filename,
        here.parent / filename,
        here.parent / "data" / filename,
        here.parent.parent / filename,
        here.parent.parent / "data" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


@st.cache_resource
def load_px_and_models():
    table_path = find_table_path("TGF05-TGH05.xls")
    if not table_path.exists():
        raise FileNotFoundError(
            "Fichier `TGF05-TGH05.xls` introuvable. Place-le à la racine du projet ou dans `data/`."
        )
    lx = load_tgf05_lx(str(table_path), sheet_name="TGF05")
    qx = build_qx_from_lx(lx)
    px_df = (1.0 - qx).clip(0.0, 1.0)

    models, metrics = load_models(MODELS_DIR)
    GEN_MIN, GEN_MAX = int(min(px_df.columns)), int(max(px_df.columns))
    return px_df, models, metrics, GEN_MIN, GEN_MAX, str(table_path)


st.set_page_config(page_title="Moteur ML", layout="wide")
st.title("Moteur ML — Comparaison Actuariel vs ML")

try:
    px_df, models, metrics, GEN_MIN, GEN_MAX, table_used = load_px_and_models()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.expander("Infos", expanded=False):
    st.write(f"Table : `{table_used}`")
    st.write(f"Générations : [{GEN_MIN}, {GEN_MAX}]")
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "target"})
    st.dataframe(metrics_df, width="stretch")

with st.sidebar:
    st.header("Paramètres du contrat")
    x = st.number_input("Âge x", min_value=0, max_value=120, value=40, step=1)
    m = st.number_input("Durée primes m", min_value=0, max_value=60, value=10, step=1)
    m_prime = st.number_input("Différé m′", min_value=0, max_value=60, value=10, step=1)
    n = st.number_input("Durée rente n", min_value=1, max_value=60, value=20, step=1)
    i = st.number_input("Taux i", min_value=0.0, max_value=0.20, value=0.02, step=0.005, format="%.3f")
    A = st.number_input("Rente annuelle A", min_value=1.0, max_value=1_000_000.0, value=1000.0, step=100.0)
    gen = st.number_input("Génération", min_value=GEN_MIN, max_value=GEN_MAX,
                          value=min(max(1985, GEN_MIN), GEN_MAX), step=1)
    t0 = st.number_input("Temps t (réserve affichée)", min_value=0, max_value=200, value=0, step=1)

if m > m_prime:
    st.error("Contrainte produit non respectée : **m ≤ m′**.")
    st.stop()

# --- Actuariel (référence)
pi1_act = float(single_premium(px_df=px_df, x=int(x), gen=int(gen), m_prime=int(m_prime), n=int(n), i=float(i), A=float(A)))
P_act   = float(annual_premium(px_df=px_df, x=int(x), gen=int(gen), m=int(m), m_prime=int(m_prime), n=int(n), i=float(i), A=float(A)))
Vt_act  = float(reserve_Vt(px_df=px_df, x=int(x), gen=int(gen), m=int(m), m_prime=int(m_prime), n=int(n), i=float(i), A=float(A), t=int(t0)))

# --- ML (prédiction)
row = {
    "x": float(x),
    "m": float(m),
    "m_prime": float(m_prime),
    "n": float(n),
    "i": float(i),
    "A": float(A),
    "generation": float(gen),
    "t": float(t0),
}
preds = predict_all(models, row)
pi1_ml = float(preds.get("single_premium", float("nan")))
P_ml   = float(preds.get("annual_premium", float("nan")))
Vt_ml  = float(preds.get("reserve", float("nan")))

st.subheader("Comparaison sur le contrat choisi")
comp = pd.DataFrame(
    [
        {"quantite": "Prime unique", "actuariel": pi1_act, "ml": pi1_ml},
        {"quantite": "Prime annuelle nivelée", "actuariel": P_act, "ml": P_ml},
        {"quantite": f"Réserve Vt (t={int(t0)})", "actuariel": Vt_act, "ml": Vt_ml},
    ]
)
comp["erreur_abs"] = (comp["ml"] - comp["actuariel"]).abs()
comp["erreur_rel"] = comp["erreur_abs"] / comp["actuariel"].replace(0, pd.NA)
st.dataframe(comp, width="stretch")

st.divider()
st.subheader("Courbe attendu vs estimé — Réserve Vt sur un contrat fixe")

T = int(m_prime + n)
ts = list(range(0, T + 1))

act_curve = []
ml_curve = []
for tt in ts:
    act_curve.append(float(reserve_Vt(px_df=px_df, x=int(x), gen=int(gen), m=int(m), m_prime=int(m_prime), n=int(n), i=float(i), A=float(A), t=int(tt))))
    row_tt = dict(row)
    row_tt["t"] = float(tt)
    ml_curve.append(float(predict_all(models, row_tt).get("reserve", float("nan"))))

df_curve = pd.DataFrame({"t": ts, "Vt_actuariel": act_curve, "Vt_ml": ml_curve})
df_curve["erreur_abs"] = (df_curve["Vt_ml"] - df_curve["Vt_actuariel"]).abs()

fig = plt.figure()
plt.plot(df_curve["t"], df_curve["Vt_actuariel"], label="Vt actuariel")
plt.plot(df_curve["t"], df_curve["Vt_ml"], label="Vt ML")
plt.xlabel("Temps t (années)")
plt.ylabel("Réserve Vt")
plt.legend()
plt.grid(True)
st.pyplot(fig)

with st.expander("Voir les valeurs", expanded=False):
    st.dataframe(df_curve, width="stretch")