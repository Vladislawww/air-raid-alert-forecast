import pandas as pd

from main import load_raw, get_region_alerts, REGION

df = load_raw()
raw = df[df["oblast"] == REGION].sort_values("started_at")
alerts = get_region_alerts(df)

print("=== Загальне ===")
print("Записів у файлі для регіону:", len(raw))
print("Унікальних тривог:", len(alerts))   # очікуємо 2267

print("\n=== Значення source ===")
print(raw["source"].value_counts())

print("\n=== Дублікати по роках ===")
by_year = pd.DataFrame({
    "всього_записів": raw.groupby(raw["started_at"].dt.year).size(),
    "зайвих_копій": raw[raw.duplicated(["started_at", "finished_at"])]
                      .groupby(raw["started_at"].dt.year).size(),
}).fillna(0).astype(int)
print(by_year)

print("\n=== Проміжки між сусідніми тривогами (після видалення дублікатів) ===")
gap_min = (alerts["started_at"] - alerts["finished_at"].shift()).dt.total_seconds() / 60
print("Перекриття (старт раніше кінця попередньої):", (gap_min < 0).sum())
print("Проміжок 0–1 хв:", ((gap_min >= 0) & (gap_min <= 1)).sum())
print("Проміжок 1–10 хв:", ((gap_min > 1) & (gap_min <= 10)).sum())

print("\n=== Унікальних тривог по місяцях ===")
monthly = alerts.set_index("started_at").resample("MS").size()
print(monthly.to_string())