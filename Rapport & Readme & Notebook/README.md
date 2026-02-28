# ActuarAI — Life Insurance Project  
### Deferred Temporary Life Annuity Pricing • Machine Learning • Streamlit App

---

## 📌 Project Overview

This project implements:

- ✅ A complete **actuarial pricing engine** based on the French mortality table **TGF05**
- ✅ A synthetic dataset generator compliant with academic specifications
- ✅ Machine Learning models (**Ridge Regression + HistGradientBoosting**)
- ✅ An interactive **Streamlit application** to compare actuarial results vs ML predictions

The objective is to demonstrate how Machine Learning can approximate deterministic actuarial pricing functions.

---

# 📦 1. Project Structure

```text
ActuarAI/
│
├── data/
│   
│
├── src/
│   ├── actuarial_engine.py
│   ├── dataset_generator.py
│   ├── ml_models.py
│   └── utils.py
│
├── app.py
├── requirements.txt
└── README.md
──TGF05.xls

```

The project was first developed in a **Jupyter Notebook (exploratory phase)** to validate formulas, test the ML pipeline, and analyze dataset behavior.  
It was then refactored into a structured **Streamlit application**.

---

# 📊 2. Actuarial Product Modeled

## Deferred Temporary Life Annuity

- Entry age: **x**
- Deferral period: **m′**
- Annual annuity amount: **A**
- Maximum payment duration: **n**
- Survival modeled via TGF05 generation table

---

## Premium Structure

- Annual premium **P**
- Paid in advance (annuity-due)
- Maximum duration **m**

---

## Core Actuarial Formulas

### Present Value of Deferred Temporary Annuity

\[
{}_{m'}a_{x:\overline{n}}
=
\sum_{k=m'}^{m'+n-1}
v^{k+1} \, {}_k p_x
\]

---

### Single Premium

\[
U = A \cdot {}_{m'}a_{x:\overline{n}}
\]

---

### Present Value of Premiums (Annuity-Due)

\[
\ddot{a}_{x:\overline{m}}
=
\sum_{k=0}^{m-1}
v^k \, {}_k p_x
\]

---

### Annual Premium

\[
P =
\frac{U}{\ddot{a}_{x:\overline{m}}}
\]

---

### Prospective Reserve

\[
V_t = PV(\text{future benefits}) - PV(\text{future premiums})
\]

---

# 🎯 3. Synthetic Dataset

Features:

- x ∈ {20,30,40,50,60}
- m ∈ {1,5,10,20,30,40}
- m′ ∈ {0,1,5,10,20,30,40}
- n ∈ {1,5,10,20,30,40,50,60}
- i ∈ {0,0.005,0.01,0.015,0.02,0.025}
- A ∈ {50,100,200,400,800,1000,2000}
- generation (year of birth)
- t (reserve evaluation time)

Targets:

- single_premium (U)
- annual_premium (P)
- reserve (V_t)

Constraint:

- Reject observation if x + m′ + n > maximum age of table

The dataset size N is configurable (1 to 1000 as required by the academic statement).

---

# 🤖 4. Machine Learning

Two models are trained independently for each target:

### Baseline Model
- Ridge Regression (with feature scaling)

### Non-linear Model
- HistGradientBoostingRegressor (HGB)

Evaluation metrics:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (coefficient of determination)

Additional analysis:

- Learning curves
- Feature importance (permutation-based)

Observations:

- HGB significantly outperforms Ridge
- Single premium is easiest to approximate
- Reserve is hardest due to regime-switch structure

---

# 🖥️ 5. Streamlit Application

The application is divided into three pages:

### 🔹 Moteur Actuariat
- Parameter input
- Exact actuarial computation
- Reserve curve visualization

### 🔹 Moteur ML
- ML prediction
- Comparison with actuarial results
- Error display (absolute & relative)

### 🔹 Performance & Training
- Dataset generation
- Model training
- Metrics display
- Learning curves
- Feature importance

This structure avoids retraining models at each use and improves clarity.

---

# 💻 6. Installation (Windows 11)

## 6.1 Prerequisites

- Windows 11
- Python 3.10+
- Internet connection

Check Python:

```powershell
python --version
```

If missing:  
Download from https://www.python.org/downloads/  
⚠️ Make sure to check **Add Python to PATH**

---

## 6.2 Navigate to Project Folder

```powershell
cd C:\Users\PC\Downloads\ActuarAI\LifeInsuranceApp
```

---

## 6.3 Create Virtual Environment

If copied from Mac:

```powershell
Remove-Item -Recurse -Force venv
```

Create new environment:

```powershell
python -m venv venv
```

Activate:

```powershell
.\venv\Scripts\activate
```

---

## 6.4 Install Dependencies

```powershell
pip install -r requirements.txt
```

If needed:

```powershell
python -m pip install streamlit pandas xlrd openpyxl scikit-learn matplotlib
```

---

## 6.5 Launch Application

Always use:

```powershell
python -m streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 6.6 Deactivate Environment

```powershell
deactivate
```

---

# 🛠 Troubleshooting

### Streamlit not recognized

```powershell
python -m streamlit run app.py
```

### Excel loading error

```powershell
pip install xlrd openpyxl
```

### Dataset too slow

Reduce N (300–500 recommended).

---

# 📈 Academic Contribution

This project demonstrates:

- The deterministic structure of actuarial pricing
- The ability of non-linear ML models to approximate actuarial functions
- The increased difficulty of approximating reserve dynamics
- The importance of model interpretability (feature importance)

It bridges **Actuarial Science and Artificial Intelligence**.

---

# 👤 Authors

**Ethan Ada & Tom Cohen**

Université Côte d’Azur — IMAFA 2025–2026