import numpy as np
import pandas as pd

from main import load_raw, get_region_alerts, build_hourly
from features import build_features, list_oblasts

df = load_raw()
data_end = df["finished_at"].max()
hourly, _ = build_hourly(get_region_alerts(df), data_end)
others = list_oblasts(df)
full_feats, _ = build_features(df, hourly, oblasts=others)

rng = np.random.default_rng(0)
cands = hourly.index[hourly.index >= pd.Timestamp("2024-01-01", tz="UTC")]
sample = rng.choice(len(cands), size=25, replace=False)

bad = 0
for i in sample:
    t = cands[i]
    d = df[df["started_at"] <= t].copy()
    d.loc[d["finished_at"] > t, "finished_at"] = t + pd.Timedelta(days=365)
    trunc, _ = build_features(d, pd.DataFrame(index=pd.DatetimeIndex([t])), oblasts=others)
    a = full_feats.loc[t].astype(float).values
    b = trunc.loc[t].astype(float).values
    same = np.isclose(a, b, equal_nan=True)
    if not same.all():
        bad += 1
        print("РОЗБІЖНІСТЬ у", t, "ознаки:", list(full_feats.columns[~same]))

print(f"\nПеревірено моментів: {len(sample)} | з розбіжністю: {bad}")
print("OK: ознаки не залежать від майбутнього" if bad == 0 else "ЗНАЙДЕНО ВИТІК - не йди далі, покажи мені")