import argparse
from pathlib import Path

import joblib
import pandas as pd

from main import load_raw, get_region_alerts
from features import build_features, list_oblasts
from pipeline import logit

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "model.joblib"
LOCAL_TZ = "Europe/Kyiv"

parser = argparse.ArgumentParser(description="Імовірність початку тривоги в Києві в наступну годину")
parser.add_argument("--time", required=True, help='Київський час, напр. "2026-09-05 18:00"')
args = parser.parse_args()

t = pd.Timestamp(args.time, tz=LOCAL_TZ).tz_convert("UTC").floor("h")

df = load_raw()
if t < pd.Timestamp("2024-01-01", tz="UTC"):
    raise SystemExit("Час занадто ранній: модель працює для дат від 2024-01-01.")
if t > df["finished_at"].max():
    raise SystemExit(f"У даних немає інформації на цей час (дані до {df['finished_at'].max()}).")

# Лише те, що відомо в момент t: тривоги, що почались пізніше, відкидаємо,
# а кінець тривоги, яка ще триває в t, ховаємо.
known = df[df["started_at"] <= t].copy()
known.loc[known["finished_at"] > t, "finished_at"] = t + pd.Timedelta(days=365)

local_str = t.tz_convert(LOCAL_TZ).strftime("%Y-%m-%d %H:%M")
alerts = get_region_alerts(known)
if ((alerts["started_at"] <= t) & (alerts["finished_at"] > t)).any():
    raise SystemExit(f"{local_str} (Київ): тривога вже триває, прогноз початку нової не має сенсу.")

bundle = joblib.load(MODEL_PATH)
feats, _ = build_features(known, pd.DataFrame(index=pd.DatetimeIndex([t])), oblasts=list_oblasts(df))
X = feats[bundle["feature_cols"]]

p_raw = bundle["model"].predict_proba(X)[:, 1]
p = float(bundle["platt"].predict_proba(logit(p_raw))[:, 1][0])

print(f"{local_str} (Київ): ймовірність початку тривоги протягом наступної години = {p:.3f} ({p * 100:.1f} %)")
print("Для порівняння: у середньому тривога починається приблизно в 6-10 % годин без тривоги.")
if str(t) < bundle["trained_until"]:
    print("Увага: цей час входить у період, на якому навчалась модель.")