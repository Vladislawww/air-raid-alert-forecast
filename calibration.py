import pandas as pd
import lightgbm as lgb

from main import load_raw, get_region_alerts, build_hourly, H
from features import build_features

TRAIN_START = pd.Timestamp("2024-01-01", tz="UTC")
N_FOLDS = 6
FOLD_MONTHS = 2

# --- 1. Дані й ознаки ---
df = load_raw()
data_end = df["finished_at"].max()
hourly, _ = build_hourly(get_region_alerts(df), data_end)
X_all, names = build_features(df, hourly)

data = X_all.join(hourly[["active_now", "target"]])
eligible = data[~data["active_now"]]

last = eligible.index.max()
t_test = last - pd.DateOffset(months=2)   # усе, що пізніше, - тест, сюди не заходимо
gap = pd.Timedelta(hours=H)

feature_cols = [c for c in X_all.columns if c != "month"]   # рішення: без month

# --- 2. Прогнози на кожному відрізку (модель не бачила цей відрізок) ---
parts = []
for k in range(N_FOLDS, 0, -1):
    v_start = t_test - pd.DateOffset(months=FOLD_MONTHS * k)
    v_end = t_test - pd.DateOffset(months=FOLD_MONTHS * (k - 1))
    val = eligible[(eligible.index >= v_start) & (eligible.index < v_end)]
    train = eligible[(eligible.index >= TRAIN_START) & (eligible.index < v_start - gap)]

    model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(train[feature_cols], train["target"])
    p = model.predict_proba(val[feature_cols])[:, 1]
    parts.append(pd.DataFrame({"p": p, "y": val["target"].astype(int).values}, index=val.index))

oof = pd.concat(parts)
print("Годин:", len(oof), "| позитивних:", int(oof["y"].sum()), "| базова частка:", round(oof["y"].mean(), 4))

# --- 3. Калібрування: 10 груп за передбаченим ризиком ---
oof["група"] = pd.qcut(oof["p"], 10, duplicates="drop")
tab = oof.groupby("група", observed=True).agg(
    годин=("y", "size"), передбачено=("p", "mean"), фактично=("y", "mean"), тривог=("y", "sum")
)
print("\n=== Калібрування (від найнижчого до найвищого ризику) ===")
print(tab.round(4).to_string())

# --- 4. Практична цінність: найризикованіші години ---
total_pos = oof["y"].sum()
print("\n=== Найризикованіші години ===")
for share in (0.05, 0.10, 0.20):
    thr = oof["p"].quantile(1 - share)
    top = oof[oof["p"] >= thr]
    print(f"Топ {int(share * 100)}% годин: частка тривог {top['y'].mean():.3f} "
          f"(у {top['y'].mean() / oof['y'].mean():.1f} раза частіше за середнє), "
          f"відловлено {top['y'].sum() / total_pos:.1%} усіх початків тривог")