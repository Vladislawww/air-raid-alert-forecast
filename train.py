import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from main import load_raw, get_region_alerts, build_hourly, H
from features import build_features

TRAIN_START = pd.Timestamp("2024-01-01", tz="UTC")

# --- 1. Дані й ознаки ---
df = load_raw()
data_end = df["finished_at"].max()
hourly, _ = build_hourly(get_region_alerts(df), data_end)
X_all, names = build_features(df, hourly)

data = X_all.join(hourly[["active_now", "target"]])
eligible = data[~data["active_now"]]

# --- 2. Поділ за часом (як у baseline.py) ---
last = eligible.index.max()
t_test = last - pd.DateOffset(months=2)
t_val = last - pd.DateOffset(months=4)
gap = pd.Timedelta(hours=H)

train = eligible[(eligible.index >= TRAIN_START) & (eligible.index < t_val - gap)]
val = eligible[(eligible.index >= t_val) & (eligible.index < t_test - gap)]
# test ПОКИ НЕ ВИКОРИСТОВУЄМО

feature_cols = list(X_all.columns)
lr_cols = [c for c in feature_cols if c not in ("hour", "dow", "month")]
y_tr, y_val = train["target"], val["target"]

print("train:", len(train), "| позитивних:", int(y_tr.sum()), "| частка:", round(y_tr.mean(), 4))
print("val  :", len(val), "| позитивних:", int(y_val.sum()), "| частка:", round(y_val.mean(), 4))

# --- 3. Моделі ---
results = {}

results["константа"] = pd.Series(y_tr.mean(), index=val.index)

hour_rate = train.groupby("hour")["target"].mean()
results["бейзлайн: година"] = val["hour"].map(hour_rate).fillna(y_tr.mean())

lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
lr.fit(train[lr_cols], y_tr)
results["логістична регресія"] = pd.Series(lr.predict_proba(val[lr_cols])[:, 1], index=val.index)

gbm = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
)
gbm.fit(train[feature_cols], y_tr)
results["LightGBM"] = pd.Series(gbm.predict_proba(val[feature_cols])[:, 1], index=val.index)

# --- 4. Оцінка на val ---
print(f"\nPR-AUC випадкового вгадувача: {y_val.mean():.4f}\n")
print(f"{'модель':<24}{'PR-AUC':<10}{'ROC-AUC':<10}{'Brier':<10}")
for name, p in results.items():
    roc = roc_auc_score(y_val, p) if p.nunique() > 1 else float("nan")
    print(f"{name:<24}{average_precision_score(y_val, p):<10.4f}{roc:<10.4f}{brier_score_loss(y_val, p):<10.4f}")

# --- 5. Найважливіші ознаки LightGBM ---
imp = pd.Series(gbm.booster_.feature_importance("gain"), index=feature_cols).sort_values(ascending=False).head(12)
print("\nТоп-12 ознак (gain):")
for col, val_ in imp.items():
    print(f"  {names.get(col, col):<45}{val_:.1f}")