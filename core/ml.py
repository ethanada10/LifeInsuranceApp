import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

FEATURES = ["x","m","m_prime","n","i","A","generation","t"]
TARGETS = ["single_premium","annual_premium","reserve"]

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def train_hgb_models(df: pd.DataFrame, seed: int = 42):
    models = {}
    metrics = {}

    X = df[FEATURES].copy()

    for target in TARGETS:
        y = df[target].copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

        model = HistGradientBoostingRegressor(random_state=seed)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        metrics[target] = {
            "MAE": float(mean_absolute_error(y_test, pred)),
            "RMSE": rmse(y_test, pred),
            "R2": float(r2_score(y_test, pred)),
        }
        models[target] = model

    return models, metrics

def save_models(models: dict, metrics: dict, models_dir: str):
    os.makedirs(models_dir, exist_ok=True)
    for target, model in models.items():
        joblib.dump(model, os.path.join(models_dir, f"hgb_{target}.pkl"))
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

def load_models(models_dir: str):
    models = {}
    for target in TARGETS:
        models[target] = joblib.load(os.path.join(models_dir, f"hgb_{target}.pkl"))
    metrics_path = os.path.join(models_dir, "metrics.json")
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    return models, metrics

def predict_all(models: dict, row) -> dict:
    """
    row peut être :
    - un dict (features -> valeurs)  ✅ cas Streamlit
    - une ligne DataFrame (1 x p)
    - une DataFrame (n x p)

    Retourne un dict {target: prediction float}.
    """
    # Si row est un dict, on le transforme en DataFrame 1 ligne
    if isinstance(row, dict):
        X = pd.DataFrame([row])
    elif isinstance(row, pd.Series):
        X = row.to_frame().T
    else:
        X = row

    # On force l'ordre des colonnes comme à l'entraînement
    X = X[FEATURES]

    return {t: float(models[t].predict(X)[0]) for t in TARGETS}

def learning_curve_hgb(df: pd.DataFrame, target: str, sizes: list, seed: int = 42) -> pd.DataFrame:
    results = []
    max_N = len(df)
    for N in sizes:
        if N > max_N:
            continue
        df_sub = df.sample(n=N, random_state=seed)
        X = df_sub[FEATURES]
        y = df_sub[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = HistGradientBoostingRegressor(random_state=seed)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        results.append({"N": N, "R2": float(r2_score(y_test, pred))})

    return pd.DataFrame(results)

def permutation_importance_df(model, X_test, y_test, seed=42, n_repeats=10) -> pd.DataFrame:
    perm = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, random_state=seed, scoring="r2")
    imp = pd.DataFrame({
        "feature": FEATURES,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std
    }).sort_values("importance_mean", ascending=False)
    return imp
