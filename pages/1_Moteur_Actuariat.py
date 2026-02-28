import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from core.actuarial import load_tgf05_lx, build_qx_from_lx, single_premium, annual_premium, reserve_Vt


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
def load_px():
    table_path = find_table_path("TGF05-TGH05.xls")
    if not table_path.exists():
        raise FileNotFoundError(
            "Fichier `TGF05-TGH05.xls` introuvable. "
            "Mets-le à la racine du projet ou dans `data/`."
        )
    lx = load_tgf05_lx(str(table_path), sheet_name="TGF05")
    qx = build_qx_from_lx(lx)
    px = (1.0 - qx).clip(0.0, 1.0)
    gen_min, gen_max = int(min(px.columns)), int(max(px.columns))
    return px, gen_min, gen_max, str(table_path)


st.set_page_config(page_title="Moteur Actuariat", layout="wide")
st.title("Moteur actuariel — Rente viagère temporaire différée (EndPay)")

st.markdown(
    """
Cette page calcule :
- **Prime unique** (π₁)
- **Prime annuelle nivelée** (P, payée en début d’année)
- - **Réserve prospective** $V_t$

Contrainte produit retenue : **m ≤ m′** (à justifier dans le rapport).
"""
)

# ---- Load px_df ----
try:
    px_df, GEN_MIN, GEN_MAX, table_used = load_px()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.expander("Infos table mortalité", expanded=False):
    st.write(f"Fichier utilisé : `{table_used}`")
    st.write(f"Générations disponibles : [{GEN_MIN}, {GEN_MAX}]")

# ---- Inputs ----
with st.sidebar:
    st.header("Paramètres du contrat")

    x = st.number_input("Âge x", min_value=0, max_value=120, value=40, step=1)
    m = st.number_input("Durée primes m", min_value=0, max_value=60, value=10, step=1)
    m_prime = st.number_input("Différé m′", min_value=0, max_value=60, value=10, step=1)
    n = st.number_input("Durée rente n", min_value=1, max_value=60, value=20, step=1)

    i = st.number_input("Taux technique i", min_value=0.0, max_value=0.20, value=0.02, step=0.005, format="%.3f")
    A = st.number_input("Montant rente annuel A", min_value=1.0, max_value=1_000_000.0, value=1000.0, step=100.0)

    gen = st.number_input(
        "Génération (année de naissance)",
        min_value=GEN_MIN,
        max_value=GEN_MAX,
        value=min(max(1985, GEN_MIN), GEN_MAX),
        step=1,
    )

    t = st.number_input("Temps t (pour Vt)", min_value=0, max_value=200, value=0, step=1)

# ---- Constraint (prof) ----
if m > m_prime:
    st.error("Contrainte produit non respectée : **m ≤ m′**. Ajuste m et/ou m′.")
    st.stop()

# ---- Compute actuarial ----
try:
    pi1 = float(single_premium(px_df=px_df, x=int(x), gen=int(gen), m_prime=int(m_prime), n=int(n), i=float(i), A=float(A)))
    P = float(annual_premium(px_df=px_df, x=int(x), gen=int(gen), m=int(m), m_prime=int(m_prime), n=int(n), i=float(i), A=float(A)))
    Vt = float(reserve_Vt(px_df=px_df, x=int(x), gen=int(gen), m=int(m), m_prime=int(m_prime), n=int(n), i=float(i), A=float(A), t=int(t)))
except Exception as e:
    st.exception(e)
    st.stop()

# ---- Display ----
c1, c2, c3 = st.columns(3)
c1.metric("Prime unique (π₁)", f"{pi1:,.6f}")
c2.metric("Prime annuelle nivelée (P)", f"{P:,.6f}")
c3.metric(f"Réserve Vₜ (t={int(t)})", f"{Vt:,.6f}")

st.divider()

# ---- Visualization (prof wants it) ----
st.subheader("Visualisation : primes et réserves sur un contrat fixe")
st.caption("Prime due tant que t < m ; réserve Vt calculée prospectivement.")

T = int(m_prime + n)
ts = list(range(0, T + 1))
premiums = [P if tt < m else 0.0 for tt in ts]
reserves = [
    float(reserve_Vt(px_df=px_df, x=int(x), gen=int(gen), m=int(m), m_prime=int(m_prime), n=int(n), i=float(i), A=float(A), t=int(tt)))
    for tt in ts
]

df = pd.DataFrame({"t": ts, "Prime annuelle P (due)": premiums, "Réserve Vt": reserves})

fig = plt.figure()
plt.plot(df["t"], df["Prime annuelle P (due)"], label="Prime annuelle (due)")
plt.plot(df["t"], df["Réserve Vt"], label="Réserve Vt")
plt.xlabel("Temps t (années)")
plt.ylabel("Montants")
plt.legend()
plt.grid(True)
st.pyplot(fig)

with st.expander("Voir le tableau", expanded=False):
    st.dataframe(df, width="stretch")