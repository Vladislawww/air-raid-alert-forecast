import numpy as np
import pandas as pd

from main import get_region_alerts, merge_intervals, REGION

LOCAL_TZ = "Europe/Kyiv"
ONE_H = np.timedelta64(1, "h")


def list_oblasts(df):
    """Список усіх областей, крім Києва, у фіксованому порядку."""
    return sorted(o for o in df["oblast"].dropna().unique() if o != REGION)


def region_intervals(df, region):
    """Початки і кінці злитих тривог області (масиви datetime64[ns])."""
    iv = merge_intervals(get_region_alerts(df, region))
    return (iv["start"].values.astype("datetime64[ns]"),
            iv["finish"].values.astype("datetime64[ns]"))


def active_at(s, f, t):
    """Чи йде тривога в момент t (початок <= t < кінець)."""
    if len(s) == 0:
        return np.zeros(len(t), dtype=bool)
    i = np.searchsorted(s, t, side="right") - 1
    return (i >= 0) & (t < f[np.clip(i, 0, None)])


def starts_in_window(s, t, hours):
    """Скільки тривог почалось у проміжку (t - hours, t]."""
    t_prev = t - np.timedelta64(hours, "h")
    return np.searchsorted(s, t, side="right") - np.searchsorted(s, t_prev, side="right")


def build_features(df, hourly, oblasts=None):
    """Ознаки для кожної години з hourly.index. Використовуються лише дані до моменту t."""
    t = hourly.index.values.astype("datetime64[ns]")
    feats, names = {}, {}

    # --- Київ: власна історія ---
    s, f = region_intervals(df, REGION)
    for w in (1, 3, 6, 24, 72):
        feats[f"kyiv_starts_{w}h"] = starts_in_window(s, t, w)

    j = np.searchsorted(f, t, side="right") - 1      # остання тривога, що ВЖЕ закінчилась (finish <= t)
    ok = j >= 0
    jj = np.clip(j, 0, None)
    feats["kyiv_hours_since_end"] = np.minimum(np.where(ok, (t - f[jj]) / ONE_H, 720.0), 720.0)
    feats["kyiv_last_dur_h"] = np.where(ok, (f[jj] - s[jj]) / ONE_H, 0.0)

    # --- Інші області (список фіксований, щоб стовпці завжди збігались) ---
    others = oblasts if oblasts is not None else list_oblasts(df)
    n_active = np.zeros(len(t), dtype=int)
    for k, o in enumerate(others):
        s, f = region_intervals(df, o)
        a = active_at(s, f, t)
        feats[f"act_o{k}"] = a.astype(int)
        feats[f"new1h_o{k}"] = starts_in_window(s, t, 1) if len(s) else np.zeros(len(t), dtype=int)
        names[f"act_o{k}"] = f"активна: {o}"
        names[f"new1h_o{k}"] = f"старт за 1г: {o}"
        n_active += a
    feats["n_active_oblasts"] = n_active

    # --- Календар (київський час) ---
    local = hourly.index.tz_convert(LOCAL_TZ)
    hour = np.asarray(local.hour)
    feats["hour"] = hour
    feats["dow"] = np.asarray(local.dayofweek)
    feats["month"] = np.asarray(local.month)
    feats["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return pd.DataFrame(feats, index=hourly.index), names