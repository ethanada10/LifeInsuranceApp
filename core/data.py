import os
import numpy as np
import pandas as pd

from core.actuarial import single_premium, annual_premium, reserve_Vt

X_VALUES  = [20, 30, 40, 50, 60]
M_VALUES  = [1, 5, 10, 20, 30, 40]
MP_VALUES = [0, 1, 5, 10, 20, 30, 40]
N_VALUES  = [1, 5, 10, 20, 30, 40, 50, 60]
I_VALUES  = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]
A_VALUES  = [50, 100, 200, 400, 800, 1000, 2000]

def sample_issue_year_for_x(rng, x: int, gen_min: int, gen_max: int, issue_year_min: int, issue_year_max: int) -> int:
    y_min = max(issue_year_min, gen_min + x)
    y_max = min(issue_year_max, gen_max + x)
    if y_min > y_max:
        return gen_max + x
    return int(rng.integers(y_min, y_max + 1))

def generate_dataset(
    N: int,
    seed: int,
    px_df: pd.DataFrame,
    gen_min: int,
    gen_max: int,
    issue_year_min: int = 1990,
    issue_year_max: int = 2025,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(N):
        x  = int(rng.choice(X_VALUES))
        m  = int(rng.choice(M_VALUES))
        mp = int(rng.choice(MP_VALUES))
        n  = int(rng.choice(N_VALUES))
        i  = float(rng.choice(I_VALUES))
        A  = float(rng.choice(A_VALUES))

        issue_year = sample_issue_year_for_x(rng, x, gen_min, gen_max, issue_year_min, issue_year_max)
        gen = issue_year - x

        t = int(rng.integers(0, mp + n + 1))

        U = single_premium(px_df, x, gen, mp, n, i, A)
        P = annual_premium(px_df, x, gen, m, mp, n, i, A)
        Vt = reserve_Vt(px_df, x, gen, m, mp, n, i, A, t)

        rows.append({
            "x": x, "m": m, "m_prime": mp, "n": n, "i": i, "A": A,
            "issue_year": issue_year, "generation": gen, "t": t,
            "single_premium": U,
            "annual_premium": P,
            "reserve": Vt
        })

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna()
    return df

def save_dataset(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
