import numpy as np
import pandas as pd
import lightgbm as lgb

from main import load_raw, get_region_alerts, build_hourly, H
from features import build_features

TRAIN_START = pd.Timestamp("2024-01-01", tz="UTC")
N_FOLDS = 6
FOLD_MONTHS = 2
GAP = pd.Timedelta(hours=H)  # відступ між train і наступним відрізком


def load_eligible():
    """Годинна таблиця (лише години без активної тривоги) з ознаками й target."""
    df = load_raw()
    data_end = df["finished_at"].max()
    hourly, _ = build_hourly(get_region_alerts(df), data_end)
    X_all, names = build_features(df, hourly)
    data = X_all.join(hourly[["active_now", "target"]])
    eligible = data[~data["active_now"]]
    feature_cols = [c for c in X_all.columns if c != "month"]  # рішення: без month
    return eligible, feature_cols, names


def make_model():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1,
    )


def test_start(eligible):
    """Початок тестового періоду (останні 2 місяці). Усе пізніше - тест."""
    return eligible.index.max() - pd.DateOffset(months=2)


def oof_predictions(eligible, feature_cols):
    """Прогнози для N_FOLDS послідовних відрізків перед тестом.
    Кожен відрізок прогнозує модель, навчена лише на даних ДО нього."""
    t_test = test_start(eligible)
    parts = []
    for n, k in enumerate(range(N_FOLDS, 0, -1), start=1):
        v_start = t_test - pd.DateOffset(months=FOLD_MONTHS * k)
        v_end = t_test - pd.DateOffset(months=FOLD_MONTHS * (k - 1))
        val = eligible[(eligible.index >= v_start) & (eligible.index < v_end)]
        train = eligible[(eligible.index >= TRAIN_START) & (eligible.index < v_start - GAP)]
        model = make_model()
        model.fit(train[feature_cols], train["target"])
        p = model.predict_proba(val[feature_cols])[:, 1]
        parts.append(pd.DataFrame(
            {"p": p, "y": val["target"].astype(int).values, "fold": n}, index=val.index))
    return pd.concat(parts)


def logit(p):
    """Перетворення ймовірності в логіт (потрібне для калібрування Platt)."""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p)).reshape(-1, 1)