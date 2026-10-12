from pathlib import Path

import numpy as np
import pandas as pd

from main import REGION
from features import region_intervals, active_at

BASE_DIR = Path(__file__).resolve().parent
OLD_PATH = BASE_DIR / "data" / "official_data_uk.csv"
NEW_PATH = BASE_DIR / "data" / "alerts_in_ua.csv"          # підстав справжню назву нового файлу
OUT_PATH = BASE_DIR / "data" / "combined_alerts.csv.gz"

# Окуповані території: у новому джерелі там постійні тривоги без кінця, це не сигнал
EXCLUDE = {"Автономна Республіка Крим", "Луганська область"}

# --- 1. Старі дані ---
old = pd.read_csv(OLD_PATH)
old["started_at"] = pd.to_datetime(old["started_at"], utc=True)
old["finished_at"] = pd.to_datetime(old["finished_at"], utc=True)
old_end = old["finished_at"].max()
old_std = old[["oblast", "started_at", "finished_at"]]
old_std = old_std[~old_std["oblast"].isin(EXCLUDE)]
print("Кінець старого файлу:", old_end)

# --- 2. Нове джерело -> та сама схема (лише повітряні тривоги) ---
new = pd.read_csv(NEW_PATH)
new = new[new["alert_type"] == "air_raid"].copy()
new["started_at"] = pd.to_datetime(new["started_at"], utc=True, format="ISO8601", errors="coerce")
new["finished_at"] = pd.to_datetime(new["finished_at"], utc=True, format="ISO8601", errors="coerce")
snapshot_end = max(new["finished_at"].max(), new["started_at"].max())
print("Кінець нового знімка:", snapshot_end)

# Незавершені тривоги: давні (понад добу) - це постійні тривоги окупованих громад, відкидаємо;
# свіжі - справжні тривоги, що тривають на момент знімка, закриваємо часом знімка
is_open = new["finished_at"].isna()
long_open = is_open & (new["started_at"] < snapshot_end - pd.Timedelta(days=1))
print(f"Незавершених тривог у новому файлі: {is_open.sum()} | давніших за добу (відкидаємо): {long_open.sum()}")
new = new[~long_open].copy()
new["finished_at"] = new["finished_at"].fillna(snapshot_end)

new["oblast"] = new["location_oblast"]
new.loc[new["location_title"] == REGION, "oblast"] = REGION
new_std = new[["oblast", "started_at", "finished_at"]].dropna()
new_std = new_std[~new_std["oblast"].isin(EXCLUDE)]

# --- 3. Перевірка узгодженості джерел у спільному проміжку: частка часу в тривозі по областях ---
lo = pd.Timestamp("2026-09-06 13:00", tz="UTC")
t = pd.date_range(lo, old_end, freq="min").values.astype("datetime64[ns]")
rows = []
for o in sorted(set(old_std["oblast"].dropna()) | set(new_std["oblast"].dropna())):
    s1, f1 = region_intervals(old_std, o)
    s2, f2 = region_intervals(new_std, o)
    rows.append((o, active_at(s1, f1, t).mean() * 100, active_at(s2, f2, t).mean() * 100))
cmp_ = pd.DataFrame(rows, columns=["область", "старий_%", "новий_%"])
cmp_["різниця"] = (cmp_["новий_%"] - cmp_["старий_%"]).round(1)
cmp_ = cmp_[(cmp_["старий_%"] > 0) | (cmp_["новий_%"] > 0)]
print(f"\nЧас у тривозі (%) у спільному проміжку {lo:%m-%d %H:%M} - {old_end:%m-%d %H:%M} UTC:")
print(cmp_.round(1).to_string(index=False))
print("Середня абсолютна різниця, в.п.:", round(cmp_["різниця"].abs().mean(), 2))

# --- 4. Склеювання: старий файл повністю + усе з нового джерела, що закінчилось після його кінця ---
combined = pd.concat([old_std, new_std[new_std["finished_at"] > old_end]], ignore_index=True)
combined["started_at"] = combined["started_at"].dt.floor("s")   # однаковий формат часу
combined["finished_at"] = combined["finished_at"].dt.floor("s")
combined = combined.sort_values("started_at")
combined.to_csv(OUT_PATH, index=False)
print(f"\nЗбережено {OUT_PATH.name}: {len(combined)} рядків, {OUT_PATH.stat().st_size / 1e6:.1f} МБ")
print("Діапазон:", combined["started_at"].min(), "→", combined["finished_at"].max())

# --- 5. Чи є різкий стрибок у місці склеювання? Тривог у Києві по тижнях ---
k = combined[combined["oblast"] == REGION].drop_duplicates(["started_at", "finished_at"])
k = k[k["started_at"] >= pd.Timestamp("2026-07-01", tz="UTC")]
weekly = k.set_index("started_at").resample("W").size()
print("\nТривог у Києві по тижнях (мітка = неділя, якою закінчується тиждень; склеювання 7 вересня):")
print(weekly.to_string())   