import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

from pipeline import load_eligible, oof_predictions

eligible, feature_cols, names = load_eligible()
oof = oof_predictions(eligible, feature_cols)

# Калібратор навчаємо на перших 4 відрізках, перевіряємо на двох останніх
fit_part = oof[oof["fold"] <= 4]
eval_part = oof[oof["fold"] >= 5]
print("Навчання калібратора:", len(fit_part), "годин,", int(fit_part["y"].sum()), "позитивних")
print("Перевірка калібратора:", len(eval_part), "годин,", int(eval_part["y"].sum()), "позитивних")


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)


# Platt: логістична регресія на логіті прогнозу
platt = LogisticRegression(C=1e6, max_iter=1000).fit(logit(fit_part["p"].values), fit_part["y"])
p_platt = platt.predict_proba(logit(eval_part["p"].values))[:, 1]

# Isotonic: ступінчаста монотонна функція
iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(fit_part["p"], fit_part["y"])
p_iso = iso.predict(eval_part["p"])

variants = {
    "без калібрування": eval_part["p"].values,
    "Platt": p_platt,
    "Isotonic": p_iso,
}
y = eval_part["y"].values

print(f"\nБазова частка у перевірочній частині: {y.mean():.4f}")
print(f"\n{'варіант':<20}{'Brier':<10}{'LogLoss':<10}{'PR-AUC':<10}{'сер. прогноз':<14}")
for name, p in variants.items():
    p_safe = np.clip(p, 1e-4, 1 - 1e-4)
    print(f"{name:<20}{brier_score_loss(y, p):<10.4f}{log_loss(y, p_safe):<10.4f}"
          f"{average_precision_score(y, p):<10.4f}{p.mean():<14.4f}")

# Таблиця калібрування на перевірочній частині: 5 груп за ризиком
print("\n=== Калібрування на перевірочній частині (5 груп) ===")
groups = pd.qcut(eval_part["p"], 5, labels=False, duplicates="drop")
tab = pd.DataFrame({
    "годин": pd.Series(y).groupby(groups.values).size(),
    "фактично": pd.Series(y).groupby(groups.values).mean(),
    "без калібр.": pd.Series(variants["без калібрування"]).groupby(groups.values).mean(),
    "Platt": pd.Series(p_platt).groupby(groups.values).mean(),
    "Isotonic": pd.Series(p_iso).groupby(groups.values).mean(),
})
print(tab.round(4).to_string())