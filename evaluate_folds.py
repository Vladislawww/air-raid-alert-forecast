import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

all_cols = list(X_all.columns)
no_month = [c for c in all_cols if c != "month"]
lr_cols = [c for c in all_cols if c not in ("hour", "dow", "month")]
kyiv_obl_col = [k for k, v in names.items() if v == "активна: Київська область"][0]


# --- 2. Моделі: кожна приймає (train, val) і повертає прогнози для val ---
def m_const(tr, va):
    return pd.Series(tr["target"].mean(), index=va.index)


def m_hour(tr, va):
    rate = tr.groupby("hour")["target"].mean()
    return va["hour"].map(rate).fillna(tr["target"].mean())


def m_kyiv_oblast(tr, va):
    """Найпростіша модель: лише чи йде тривога в Київській області."""
    rate = tr.groupby(kyiv_obl_col)["target"].mean()
    return va[kyiv_obl_col].map(rate).fillna(tr["target"].mean())


def m_logreg(tr, va):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    m.fit(tr[lr_cols], tr["target"])
    return pd.Series(m.predict_proba(va[lr_cols])[:, 1], index=va.index)


def make_lgbm(cols):
    def f(tr, va):
        m = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        m.fit(tr[cols], tr["target"])
        return pd.Series(m.predict_proba(va[cols])[:, 1], index=va.index)
    return f


MODELS = {
    "константа": m_const,
    "година": m_hour,
    "лише Київська обл.": m_kyiv_oblast,
    "логістична регресія": m_logreg,
    "LightGBM (з month)": make_lgbm(all_cols),
    "LightGBM (без month)": make_lgbm(no_month),
}

# --- 3. Walk-forward: кожне вікно оцінюється моделлю, навченою лише на даних ДО нього ---
rows = []
for k in range(N_FOLDS, 0, -1):
    v_start = t_test - pd.DateOffset(months=FOLD_MONTHS * k)
    v_end = t_test - pd.DateOffset(months=FOLD_MONTHS * (k - 1))
    val = eligible[(eligible.index >= v_start) & (eligible.index < v_end)]
    train = eligible[(eligible.index >= TRAIN_START) & (eligible.index < v_start - gap)]
    base = val["target"].mean()
    label = v_start.strftime("%Y-%m")
    for name, fn in MODELS.items():
        p = fn(train, val)
        rows.append({
            "вікно": label, "модель": name, "позитивних": int(val["target"].sum()),
            "base": base,
            "PR_AUC": average_precision_score(val["target"], p),
            "Brier": brier_score_loss(val["target"], p),
        })

res = pd.DataFrame(rows)
res["lift"] = res["PR_AUC"] / res["base"]   # у скільки разів краще за випадкове вгадування

order = list(MODELS.keys())
print("=== Частка позитивних і кількість у кожному вікні ===")
print(res[res["модель"] == "константа"][["вікно", "позитивних", "base"]].round(4).to_string(index=False))

print("\n=== PR-AUC по вікнах ===")
print(res.pivot(index="вікно", columns="модель", values="PR_AUC")[order].round(3).to_string())

print("\n=== Lift (PR-AUC / випадковий) по вікнах ===")
print(res.pivot(index="вікно", columns="модель", values="lift")[order].round(2).to_string())

print("\n=== Brier по вікнах (нижче = краще) ===")
print(res.pivot(index="вікно", columns="модель", values="Brier")[order].round(4).to_string())

print("\n=== Середнє по вікнах ===")
summary = res.groupby("модель")[["PR_AUC", "lift", "Brier"]].mean().loc[order].round(4)
print(summary.to_string())