import numpy as np
import pandas as pd

H = 1  # горизонт прогнозу, годин

def load_kyiv(path="data/official_data_uk.csv"):
    df = pd.read_csv(path)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)
    data_end = df["finished_at"].max()          # кінець спостережень у ВСЬОМУ файлі
    kyiv = df[df["oblast"] == "м. Київ"].sort_values("started_at")
    return kyiv, data_end

def merge_intervals(kyiv):
    """Зливає тривоги, що перекриваються, в безперервні інтервали."""
    merged = []
    for s, f in zip(kyiv["started_at"], kyiv["finished_at"]):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], f)
        else:
            merged.append([s, f])
    return pd.DataFrame(merged, columns=["start", "finish"])

def build_hourly(kyiv, data_end, h=H):
    iv = merge_intervals(kyiv)
    starts = iv["start"].values.astype("datetime64[ns]")
    finishes = iv["finish"].values.astype("datetime64[ns]")

    t_first = kyiv["started_at"].min().floor("h")
    t_last = data_end.floor("h") - pd.Timedelta(hours=h)   # щоб ціль була повністю спостережена
    grid = pd.date_range(t_first, t_last, freq="h", tz="UTC")
    t = grid.values.astype("datetime64[ns]")

    # чи йде тривога в момент t
    i = np.searchsorted(starts, t, side="right") - 1
    active = (i >= 0) & (t < finishes[np.clip(i, 0, None)])

    # чи почнеться нова тривога в (t, t+h]
    t_h = t + np.timedelta64(h, "h")
    n_start = np.searchsorted(starts, t_h, side="right") - np.searchsorted(starts, t, side="right")
    target = n_start > 0

    out = pd.DataFrame({"active_now": active, "target": target}, index=grid)
    return out, iv

if __name__ == "__main__":
    kyiv, data_end = load_kyiv()
    hourly, iv = build_hourly(kyiv, data_end)
    print("Кількість окремих тривог до злиття:", len(kyiv), "→ після злиття:", len(iv))
    print("Годин у сітці:", len(hourly))
    print("Частка годин із активною тривогою:", hourly["active_now"].mean().round(3))

    eligible = hourly[~hourly["active_now"]]
    print("Придатних рядків (без активної тривоги):", len(eligible))
    print("Частка позитивного класу (target=1):", eligible["target"].mean().round(4))
    print(eligible.index.min(), eligible.index.max())