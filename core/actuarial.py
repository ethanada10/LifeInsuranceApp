import numpy as np
import pandas as pd

def v(i: float) -> float:
    return 1.0 / (1.0 + i)

def load_tgf05_lx(path: str, sheet_name: str = "TGF05") -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)

    header = raw.iloc[1].tolist()
    header[0] = "Age"

    df = raw.iloc[2:].copy()
    df.columns = header
    df = df.dropna(subset=["Age"])
    df["Age"] = df["Age"].astype(int)
    df = df.set_index("Age")

    df.columns = [int(c) for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    return df

def build_qx_from_lx(lx_df: pd.DataFrame) -> pd.DataFrame:
    qx_df = (lx_df - lx_df.shift(-1)) / lx_df
    last_age = qx_df.index.max()
    qx_df.loc[last_age] = np.where(lx_df.loc[last_age] > 0, 1.0, 0.0)
    return qx_df.clip(0.0, 1.0)

def tpx(px_df: pd.DataFrame, age: int, gen: int, t: int) -> float:
    if t <= 0:
        return 1.0
    if gen not in px_df.columns:
        return 0.0
    if age not in px_df.index or (age + t - 1) not in px_df.index:
        return 0.0

    p = 1.0
    for a in range(age, age + t):
        p *= float(px_df.loc[a, gen])
    return p

def deferred_temp_annuity_immediate(px_df: pd.DataFrame, x: int, gen: int, m_prime: int, n: int, i: float) -> float:
    vv = v(i)
    s = 0.0
    for k in range(m_prime, m_prime + n):
        s += (vv ** (k + 1)) * tpx(px_df, x, gen, k)
    return s

def single_premium(px_df: pd.DataFrame, x: int, gen: int, m_prime: int, n: int, i: float, A: float) -> float:
    return A * deferred_temp_annuity_immediate(px_df, x, gen, m_prime, n, i)

def annuity_due_temp(px_df: pd.DataFrame, x: int, gen: int, m: int, i: float) -> float:
    vv = v(i)
    s = 0.0
    for j in range(0, m):
        s += (vv ** j) * tpx(px_df, x, gen, j)
    return s

def annual_premium(px_df: pd.DataFrame, x: int, gen: int, m: int, m_prime: int, n: int, i: float, A: float) -> float:
    U = single_premium(px_df, x, gen, m_prime, n, i, A)
    denom = annuity_due_temp(px_df, x, gen, m, i)
    return U / denom if denom > 0 else np.nan

def reserve_Vt(px_df: pd.DataFrame, x: int, gen: int, m: int, m_prime: int, n: int, i: float, A: float, t: int) -> float:
    vv = v(i)

    # PV prestations futures
    start_k = max(m_prime, t)
    end_k = m_prime + n - 1

    pv_b = 0.0
    for k in range(start_k, end_k + 1):
        pv_b += (vv ** ((k + 1) - t)) * tpx(px_df, x + t, gen, k - t)
    pv_b *= A

    # PV primes futures
    pv_p = 0.0
    if t < m:
        for j in range(t, m):
            pv_p += (vv ** (j - t)) * tpx(px_df, x + t, gen, j - t)
        P = annual_premium(px_df, x, gen, m, m_prime, n, i, A)
        pv_p *= (P if np.isfinite(P) else 0.0)

    return pv_b - pv_p
