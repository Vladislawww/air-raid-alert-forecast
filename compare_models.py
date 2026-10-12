import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline import (load_eligible, make_model, test_start,
                      TRAIN_START, GAP, N_FOLDS, FOLD_MONTHS)

eligible, feature_cols, names = load_eligible()
lr_cols = [c for c in feature_cols if c not in ("hour", "dow")]
t_test = test_start(eligible)          # усе після цього - тест, сюди не заходимо


def fit_predict(kind, tr, va):
    if kind == "логрегресія":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        m.fit(tr[lr_cols], tr["target"])
        return m.predict_proba(va[lr_cols])[:, 1]
    if kind == "LightGBM":
        m = make_model().fit(tr[feature_cols], tr["target"])
        return m.predict_proba(va[feature_cols])[:, 1]
    if kind == "HistGradBoosting":
        m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                           min_samples_leaf=50, l2_regularization=1.0, random_state=42)
        m.fit(tr[feature_cols], tr["target"])
        return m.predict_proba(va[feature_cols])[:, 1]
    if kind == "RandomForest":
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=30, max_features="sqrt",
                                   n_jobs=-1, random_state=42)
        m.fit(tr[feature_cols], tr["target"])
        return m.predict_proba(va[feature_cols])[:, 1]


kinds = ["логрегресія", "LightGBM", "HistGradBoosting", "RandomForest"]
lifts = {k: [] for k in kinds + ["ансамбль лог+LGBM"]}
labels = []

for k in range(N_FOLDS, 0, -1):
    v_start = t_test - pd.DateOffset(months=FOLD_MONTHS * k)
    v_end = t_test - pd.DateOffset(months=FOLD_MONTHS * (k - 1))
    va = eligible[(eligible.index >= v_start) & (eligible.index < v_end)]
    tr = eligible[(eligible.index >= TRAIN_START) & (eligible.index < v_start - GAP)]
    y = va["target"].astype(int).values
    preds = {kind: fit_predict(kind, tr, va) for kind in kinds}
    preds["ансамбль лог+LGBM"] = (preds["логрегресія"] + preds["LightGBM"]) / 2
    for name, p in preds.items():
        lifts[name].append(average_precision_score(y, p) / y.mean())
    labels.append(v_start.strftime("%Y-%m"))

res = pd.DataFrame(lifts, index=labels)
print("=== Lift по відрізках (PR-AUC / випадковий) ===")
print(res.round(2).to_string())
print("\n=== Середній lift ===")
print(res.mean().round(3).to_string())
print("\n=== У скількох із 6 відрізків краща за LightGBM ===")
for c in res.columns:
    if c != "LightGBM":
        print(f"{c:<20}{int((res[c] > res['LightGBM']).sum())} з {len(res)}")
print("\nПравило: замінюємо LightGBM лише якщо модель краща у ≥5 з 6 і середній lift вищий більш ніж на 0.15.")