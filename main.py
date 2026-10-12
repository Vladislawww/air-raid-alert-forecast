from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "official_data_uk.csv"
REGION = "м. Київ"
H = 1  # горизонт прогнозу, годин


def load_raw(path=DATA_PATH):
    """Завантажує весь файл і перетворює час у datetime (UTC)."""
    df = pd.read_csv(path)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    return df


def get_region_alerts(df, region=REGION):
    """Тривоги одного регіону без точних дублікатів, відсортовані за часом."""
    alerts = (
        df[df["oblast"] == region]
        .drop_duplicates(["started_at", "finished_at"])
        .sort_values("started_at")
        .reset_index(drop=True)
    )
    return alerts


def merge_intervals(alerts):
    """Зливає тривоги, що перекриваються, в безперервні інтервали."""
    merged = []
    for s, f in zip(alerts["started_at"], alerts["finished_at"]):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], f)
        else:
            merged.append([s, f])
    return pd.DataFrame(merged, columns=["start", "finish"])


def build_hourly(alerts, data_end, h=H):
    """Таблиця: година t -> active_now (чи йде тривога) і target (чи почнеться нова за (t, t+h])."""
    iv = merge_intervals(alerts)
    starts = iv["start"].values.astype("datetime64[ns]")
    finishes = iv["finish"].values.astype("datetime64[ns]")

    t_first = alerts["started_at"].min().floor("h")
    t_last = data_end.floor("h") - pd.Timedelta(hours=h)  # щоб ціль була повністю спостережена
    grid = pd.date_range(t_first, t_last, freq="h", tz="UTC")
    t = grid.values.astype("datetime64[ns]")

    # чи йде тривога в момент t
    i = np.searchsorted(starts, t, side="right") - 1
    active = (i >= 0) & (t < finishes[np.clip(i, 0, None)])

    # чи почнеться нова тривога в (t, t+h]
    t_h = t + np.timedelta64(h, "h")
    n_start = np.searchsorted(starts, t_h, side="right") - np.searchsorted(starts, t, side="right")
    target = n_start > 0

    hourly = pd.DataFrame({"active_now": active, "target": target}, index=grid)
    return hourly, iv


if __name__ == "__main__":
    df = load_raw()
    data_end = df["finished_at"].max()  # кінець спостережень у ВСЬОМУ файлі
    alerts = get_region_alerts(df)
    hourly, iv = build_hourly(alerts, data_end)

    print("Унікальних тривог (після видалення дублікатів):", len(alerts))
    print("Інтервалів після злиття:", len(iv))
    print("Годин у сітці:", len(hourly))
    print("Частка годин із активною тривогою:", round(hourly["active_now"].mean(), 3))

    eligible = hourly[~hourly["active_now"]]
    print("Придатних рядків (без активної тривоги):", len(eligible))
    print("Частка позитивного класу (target=1):", round(eligible["target"].mean(), 4))
    print("Діапазон:", eligible.index.min(), "→", eligible.index.max())