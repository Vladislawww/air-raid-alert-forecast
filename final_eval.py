from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from pipeline import (load_eligible, oof_predictions, make_model, test_start,
                      TRAIN_START, GAP, logit)

eligible, feature_cols, names = load_eligible()
t_test = test_start(eligible)

# 1. Калібратор: навчаємо на прогнозах усіх 6 відрізків ДО тесту
oof = oof_predictions(eligible, feature_cols)
platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(oof["p"].values), oof["y"])

# 2. Фінальна модель: усе до тесту (з відступом GAP)
train = eligible[(eligible.index >= TRAIN_START) & (eligible.index < t_test - GAP)]
test = eligible[eligible.index >= t_test]
model = make_model().fit(train[feature_cols], train["target"])

p_raw = model.predict_proba(test[feature_cols])[:, 1]
p_cal = platt.predict_proba(logit(p_raw))[:, 1]
y = test["target"].astype(int).values

# 3. Бейзлайни (усе, що потрібне для них, береться з train)
kyiv_obl_col = [k for k, v in names.items() if v == "активна: Київська область"][0]
const = np.full(len(test), train["target"].mean())
hour_rate = train.groupby("hour")["target"].mean()
by_hour = test["hour"].map(hour_rate).fillna(train["target"].mean()).values
obl_rate = train.groupby(kyiv_obl_col)["target"].mean()
by_obl = test[kyiv_obl_col].map(obl_rate).fillna(train["target"].mean()).values

print(f"TEST: {test.index.min()} → {test.index.max()}")
print(f"Годин: {len(test)} | позитивних: {y.sum()} | базова частка: {y.mean():.4f} "
      f"(у train: {train['target'].mean():.4f})\n")

print(f"{'модель':<26}{'PR-AUC':<10}{'lift':<8}{'Brier':<10}{'LogLoss':<10}{'сер. прогноз':<12}")
for name, p in [("константа", const), ("година", by_hour), ("лише Київська обл.", by_obl),
                ("LightGBM (без калібр.)", p_raw), ("LightGBM + Platt", p_cal)]:
    ap = average_precision_score(y, p)
    print(f"{name:<26}{ap:<10.4f}{ap / y.mean():<8.2f}{brier_score_loss(y, p):<10.4f}"
          f"{log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)):<10.4f}{p.mean():<12.4f}")

# 4. Калібрування на тесті
print("\n=== Калібрування на тесті (5 груп за ризиком) ===")
g = pd.qcut(pd.Series(p_cal), 5, labels=False, duplicates="drop")
tab = pd.DataFrame({"годин": pd.Series(y).groupby(g).size(),
                    "передбачено": pd.Series(p_cal).groupby(g).mean(),
                    "фактично": pd.Series(y).groupby(g).mean()})
print(tab.round(4).to_string())

# 5. Найризикованіші години
print("\n=== Найризикованіші години ===")
for share in (0.05, 0.10, 0.20):
    thr = np.quantile(p_cal, 1 - share)
    m = p_cal >= thr
    print(f"Топ {int(share * 100)}%: частка тривог {y[m].mean():.3f} "
          f"({y[m].mean() / y.mean():.1f}x від середнього), відловлено {y[m].sum() / y.sum():.1%}")

# 6. Зберігаємо модель для predict.py
Path("artifacts").mkdir(exist_ok=True)
joblib.dump({"model": model, "platt": platt, "feature_cols": feature_cols,
             "trained_until": str(t_test)}, "artifacts/model.joblib")
print("\nЗбережено artifacts/model.joblib")