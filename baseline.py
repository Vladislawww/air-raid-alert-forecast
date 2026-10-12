import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from main import load_raw, get_region_alerts, build_hourly, H

# --- 1. Таблиця годин (як у main.py) ---
df = load_raw()
data_end = df["finished_at"].max()
alerts = get_region_alerts(df)
hourly, _ = build_hourly(alerts, data_end)

eligible = hourly[~hourly["active_now"]].copy()
local = eligible.index.tz_convert("Europe/Kyiv")
eligible["hour"] = local.hour
eligible["dow"] = local.dayofweek

# --- 2. Поділ за часом: validation і test не залежать від вибору train ---
last = eligible.index.max()
t_test = last - pd.DateOffset(months=2)
t_val = last - pd.DateOffset(months=4)
gap = pd.Timedelta(hours=H)  # відступ, щоб ціль train не зазирала у val

val = eligible[(eligible.index >= t_val) & (eligible.index < t_test - gap)]
test = eligible[eligible.index >= t_test]  # ПОКИ НЕ ВИКОРИСТОВУЄМО

print("val :", val.index.min(), "→", val.index.max(), "| рядків:", len(val),
      "| частка target:", round(val["target"].mean(), 4))
print("test:", test.index.min(), "→", test.index.max(), "| рядків:", len(test),
      "| частка target:", round(test["target"].mean(), 4))


# --- 3. Бейзлайни ---
def predict(train, val, keys):
    """Частота target у train за ключами keys; для невідомих комбінацій - загальна частка."""
    overall = train["target"].mean()
    if not keys:
        return pd.Series(overall, index=val.index)
    rate = train.groupby(keys)["target"].mean().rename("p")
    return val.join(rate, on=keys)["p"].fillna(overall)


def evaluate(p, y):
    return round(average_precision_score(y, p), 4), round(brier_score_loss(y, p), 4)


print("\nPR-AUC випадкового вгадувача (= частка target у val):", round(val["target"].mean(), 4))
print()
print(f"{'train від':<12}{'бейзлайн':<16}{'рядків train':<14}{'PR-AUC':<10}{'Brier':<10}")

for start in ["2022-03-01", "2023-07-01", "2024-01-01", "2024-07-01"]:
    train = eligible[(eligible.index >= pd.Timestamp(start, tz="UTC"))
                     & (eligible.index < t_val - gap)]
    for name, keys in [("константа", []), ("година", ["hour"]), ("година×день", ["hour", "dow"])]:
        p = predict(train, val, keys)
        pr, br = evaluate(p, val["target"])
        print(f"{start:<12}{name:<16}{len(train):<14}{pr:<10}{br:<10}")