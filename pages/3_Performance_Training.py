import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

from core.actuarial import load_tgf05_lx, build_qx_from_lx
from core.data import generate_dataset, save_dataset, load_dataset
from core.ml import (
    train_hgb_models,
    save_models,
    learning_curve_hgb,
    FEATURES,
    TARGETS,
    permutation_importance_df,
)

TABLE_FILENAME = "TGF05-TGH05.xls"
DATA_PATH = "data/dataset.csv"
MODELS_DIR = "models"


def find_table_path(filename: str = TABLE_FILENAME) -> Path:
    """
    Cherche le fichier Excel de mortalité de manière PORTABLE (sans chemin Mac absolu).
    """
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
def load_qx():
    table_path = find_table_path(TABLE_FILENAME)
    if not table_path.exists():
        raise FileNotFoundError(
            "Fichier table mortalité introuvable (TGF05-TGH05.xls). "
            "Place-le à la racine du projet ou dans un dossier `data/`."
        )
    lx = load_tgf05_lx(str(table_path), "TGF05")
    qx = build_qx_from_lx(lx)
    gen_min, gen_max = int(min(lx.columns)), int(max(lx.columns))
    return qx, gen_min, gen_max, str(table_path)


st.set_page_config(page_title="Performance & Training", layout="wide")
st.title("Performance & Training — Dataset, entraînement, diagnostics")

# ------------------------
# Load mortality
# ------------------------
try:
    qx, GEN_MIN, GEN_MAX, table_path_used = load_qx()
except Exception as e:
    st.error(str(e))
    st.stop()

with st.expander("Infos chargement", expanded=False):
    st.write(f"Table mortalité utilisée : `{table_path_used}`")
    st.write(f"Générations disponibles : [{GEN_MIN}, {GEN_MAX}]")
    st.write(f"Dataset : `{DATA_PATH}` — Modèles : `{MODELS_DIR}/`")

st.markdown(
    """
Objectif : documenter une **méthodologie de validation** (pas seulement un R²).
On affiche :
- métriques (MAE, RMSE, R²),
- **scatter attendu vs prédit**,
- résidus,
- learning curves (avec prudence),
- importance des variables (permutation importance).
"""
)

# ------------------------
# Dataset generation / loading
# ------------------------
st.subheader("1) Dataset")

colA, colB, colC = st.columns(3)
with colA:
    N = st.number_input("Taille dataset à générer", min_value=200, max_value=200_000, value=5000, step=500)
with colB:
    seed = st.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)
with colC:
    regenerate = st.checkbox("Régénérer le dataset (écrase dataset.csv)", value=False)

if regenerate:
    df = generate_dataset(qx=qx, n_samples=int(N), seed=int(seed), gen_min=GEN_MIN, gen_max=GEN_MAX)
    save_dataset(df, DATA_PATH)
    st.success(f"Dataset régénéré et sauvegardé dans `{DATA_PATH}`")
else:
    try:
        df = load_dataset(DATA_PATH)
    except Exception:
        st.warning("Dataset introuvable : coche la case pour le générer (ci-dessus).")
        st.stop()

st.write("Aperçu :")
st.dataframe(df.head(20), width="stretch")

# ------------------------
# Quick constraint checks
# ------------------------
st.markdown("**Contrôles rapides (contraintes & cohérence)** :")

checks = {
    "m <= m_prime": float((df["m"] <= df["m_prime"]).mean()) if "m" in df and "m_prime" in df else None,
    "t <= m_prime + n": float((df["t"] <= (df["m_prime"] + df["n"])).mean()) if all(k in df for k in ["t", "m_prime", "n"]) else None,
    "x + m_prime + n <= 120": float((df["x"] + df["m_prime"] + df["n"] <= 120).mean()) if all(k in df for k in ["x", "m_prime", "n"]) else None,
}
st.dataframe(pd.DataFrame([checks]), width="stretch")

# ------------------------
# Training HGB baseline
# ------------------------
st.subheader("2) Entraînement HGB (baseline)")

if st.button("Entraîner / réentraîner les modèles HGB"):
    models, metrics = train_hgb_models(df, seed=int(seed))
    save_models(models, metrics, MODELS_DIR)
    st.success("Modèles entraînés et sauvegardés.")
else:
    st.info("Clique sur le bouton pour entraîner (ou réentraîner) les modèles.")

# ------------------------
# Load metrics if available
# ------------------------
import os
import json

metrics_path = os.path.join(MODELS_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    metrics_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "target"})
    st.write("Métriques (holdout 20%) :")
    st.dataframe(metrics_df, width="stretch")
else:
    st.warning("Aucune métrique trouvée (metrics.json). Entraîne les modèles d'abord.")
    st.stop()

# ------------------------
# Diagnostics
# ------------------------
st.subheader("3) Diagnostics — attendu vs prédit")

target = st.selectbox("Cible à diagnostiquer", TARGETS, index=0)

X = df[FEATURES].copy()
y = df[target].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=int(seed))

model = HistGradientBoostingRegressor(random_state=int(seed))
model.fit(X_train, y_train)
pred = model.predict(X_test)

diag = pd.DataFrame({"y_true": y_test.values, "y_pred": pred})
diag["residual"] = diag["y_pred"] - diag["y_true"]
diag["abs_err"] = diag["residual"].abs()

# ---- METRICS (FIX sklearn: no squared=False) ----
c1, c2, c3 = st.columns(3)
c1.metric("MAE", f"{mean_absolute_error(y_test, pred):.6f}")

rmse = (mean_squared_error(y_test, pred) ** 0.5)
c2.metric("RMSE", f"{rmse:.6f}")

c3.metric("R²", f"{r2_score(y_test, pred):.4f}")

# Scatter plot
fig1 = plt.figure()
plt.scatter(diag["y_true"], diag["y_pred"], s=8)
plt.xlabel("Attendu (actuariel)")
plt.ylabel("Prédit (ML)")
plt.title(f"Attendu vs prédit — {target}")
plt.grid(True)
st.pyplot(fig1)

# Residuals plot
fig2 = plt.figure()
plt.scatter(diag["y_true"], diag["residual"], s=8)
plt.axhline(0)
plt.xlabel("Attendu (actuariel)")
plt.ylabel("Résidu (prédit - attendu)")
plt.title(f"Résidus — {target}")
plt.grid(True)
st.pyplot(fig2)

with st.expander("Table diagnostics (échantillon)", expanded=False):
    st.dataframe(diag.sample(n=min(200, len(diag)), random_state=int(seed)), width="stretch")

# ------------------------
# Learning curves
# ------------------------
st.subheader("4) Learning curves (prudence)")

sizes = st.multiselect(
    "Tailles d'échantillon (N)",
    options=[200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    default=[500, 1000, 2000, 5000, 10000],
)

if sizes:
    lc = learning_curve_hgb(df, target=target, sizes=sorted(sizes), seed=int(seed))
    st.dataframe(lc, width="stretch")

    fig3 = plt.figure()
    plt.plot(lc["N"], lc["R2"], marker="o")
    plt.xlabel("Taille N")
    plt.ylabel("R² (holdout)")
    plt.title(f"Learning curve — {target}")
    plt.grid(True)
    st.pyplot(fig3)

    st.caption("Attention : avec peu de points, on décrit une tendance, sans conclure à une convergence ferme.")
else:
    st.info("Sélectionne au moins une taille pour afficher la learning curve.")

# ------------------------
# Permutation importance
# ------------------------
st.subheader("5) Importance des variables (permutation importance)")

imp = permutation_importance_df(model, X_test, y_test, seed=int(seed), n_repeats=10)
st.dataframe(imp, width="stretch")

fig4 = plt.figure()
plt.barh(imp["feature"], imp["importance_mean"])
plt.xlabel("Importance moyenne (ΔR²)")
plt.title(f"Permutation importance — {target}")
plt.grid(True, axis="x")
st.pyplot(fig4)