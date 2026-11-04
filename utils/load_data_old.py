import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional
from gluonts.dataset.repository.datasets import get_dataset


# ============================================================
# 1. Dataset
# ============================================================

class ForecastingWindowDataset(Dataset):
    """
    Pure forecasting dataset.

    Each sample:
        x : (L, C) historical window
        y : (H, C) future targets
    """

    def __init__(self, x_list: List[np.ndarray], y_list: List[np.ndarray]):
        assert len(x_list) == len(y_list), "x_list and y_list must have same length"
        self.x_list = x_list
        self.y_list = y_list

    def __len__(self):
        return len(self.x_list)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x_list[idx]).float()
        y = torch.from_numpy(self.y_list[idx]).float()
        return x, y


# ============================================================
# 2. Collate Function
# ============================================================

def forecasting_collate(batch):
    """
    batch: list of (x, y)
        x: (L_i, C)
        y: (H, C)
    returns:
        x_batch : (B, L_max, C)
        y_batch : (B, H, C)
        lengths : (B,)
    """
    xs, ys = zip(*batch)

    B = len(xs)
    C = xs[0].shape[1]
    H = ys[0].shape[0]

    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    L_max = lengths.max().item()

    x_batch = torch.zeros(B, L_max, C, dtype=torch.float32)
    y_batch = torch.zeros(B, H, C, dtype=torch.float32)

    for i, (x, y) in enumerate(zip(xs, ys)):
        x_batch[i, :x.shape[0]] = x
        y_batch[i] = y

    return x_batch, y_batch, lengths


# ============================================================
# 3. Sliding Window Construction
# ============================================================

def build_forecasting_windows(
    data: np.ndarray,
    seq_len: int,
    pred_len: int,
    stride: int,
    train_range: Tuple[int, int],
    val_range: Tuple[int, int],
    test_range: Tuple[int, int],
):
    """
    Construct forecasting windows for train / val / test splits.

    data: (T, C)
    """
    splits = {
        "train": train_range,
        "val": val_range,
        "test": test_range,
    }
    results = {}

    for split, (start, end) in splits.items():
        x_list, y_list = [], []

        # last valid start t must satisfy:
        # t + seq_len + pred_len <= end
        max_t = end - seq_len - pred_len + 1

        if max_t <= start:
            results[split] = (x_list, y_list)
            continue

        for t in range(start, max_t, stride):
            x = data[t:t + seq_len]                               # (seq_len, C)
            y = data[t + seq_len:t + seq_len + pred_len]         # (pred_len, C)
            x_list.append(x.astype(np.float32))
            y_list.append(y.astype(np.float32))

        results[split] = (x_list, y_list)

    return results


# ============================================================
# 4. Time Feature Utilities
# ============================================================

def build_time_features(dt_index: pd.DatetimeIndex | pd.Series) -> np.ndarray:
    """
    Build standard time features used in forecasting literature.
    Handles both DatetimeIndex and Series[datetime64].
    """
    if isinstance(dt_index, pd.DatetimeIndex):
        hours = dt_index.hour
        dayofweek = dt_index.dayofweek
        day = dt_index.day
        dayofyear = dt_index.dayofyear
        month = dt_index.month
    else:
        dt_index = pd.to_datetime(dt_index)
        hours = dt_index.dt.hour
        dayofweek = dt_index.dt.dayofweek
        day = dt_index.dt.day
        dayofyear = dt_index.dt.dayofyear
        month = dt_index.dt.month

    return np.stack(
        [
            np.asarray(hours, dtype=np.float32) / 23.0,
            np.asarray(dayofweek, dtype=np.float32) / 6.0,
            np.asarray(day, dtype=np.float32) / 31.0,
            np.asarray(dayofyear, dtype=np.float32) / 365.0,
            np.asarray(month, dtype=np.float32) / 12.0,
        ],
        axis=1,
    ).astype(np.float32)


def build_time_mark_from_date(date_series: pd.Series, freq: str = "h") -> np.ndarray:
    dt = pd.to_datetime(date_series)

    month = (dt.dt.month.values - 1).astype(np.int64)
    day = (dt.dt.day.values - 1).astype(np.int64)
    weekday = dt.dt.weekday.values.astype(np.int64)
    hour = dt.dt.hour.values.astype(np.int64)

    if freq in ["h", "15min"]:
        time_mark = np.stack([month, day, weekday, hour], axis=1)
    else:
        raise NotImplementedError(f"freq={freq} is not supported yet.")

    return time_mark.astype(np.float32)


# ============================================================
# 5. Preprocess Functions
#    注意：这里不再在整个数据集上 fit scaler
#    统一返回 raw data，scaler 在 load_forecasting_data 里只用 train 拟合
# ============================================================

def preprocess_csv(
    path: str,
    time_col: str = "date",
    drop_cols: Optional[List[str]] = None,
    scale: bool = False,
    scaler: Optional[StandardScaler] = None,
):
    """
    Generic CSV preprocessor for forecasting.

    Important:
    - 默认不在这里做全数据 fit_transform，避免数据泄露
    - 推荐在 load_forecasting_data 中仅用 train split 来 fit scaler
    """
    df = pd.read_csv(path)

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)

    if drop_cols is not None:
        df = df.drop(columns=drop_cols)

    dt_index = df[time_col]
    data = df.drop(columns=[time_col]).values.astype(np.float32)

    time_feat = build_time_features(dt_index)

    if not scale:
        return data, time_feat, None

    if scaler is None:
        raise ValueError("When scale=True in preprocess_csv, scaler must be provided.")

    data_scaled = scaler.transform(data).astype(np.float32)
    return data_scaled, time_feat, scaler


def preprocess_solar(
    scale: bool = False,
    scaler: Optional[StandardScaler] = None,
):
    """
    GluonTS solar_nips -> (T, C)
    """
    dataset = get_dataset("solar_nips")
    freq = dataset.metadata.freq

    series_list = []
    start_times = []

    for entry in dataset.train:
        target = np.asarray(entry["target"], dtype=np.float32)   # (T,)
        series_list.append(target)
        start_times.append(entry["start"].to_timestamp())

    lengths = [len(s) for s in series_list]
    assert len(set(lengths)) == 1, "Solar series lengths are not aligned!"

    T = lengths[0]
    C = len(series_list)
    print(f"[preprocess_solar] Loaded {C} series, length {T} each.")

    data_matrix = np.stack(series_list, axis=1).astype(np.float32)   # (T, C)

    time_index = pd.date_range(
        start=start_times[0],
        periods=T,
        freq=freq
    )

    time_mark = np.stack(
        [
            time_index.hour,
            time_index.day,
            time_index.weekday,
            time_index.month,
        ],
        axis=1
    ).astype(np.float32)

    if not scale:
        return data_matrix, time_mark, None

    if scaler is None:
        raise ValueError("When scale=True in preprocess_solar, scaler must be provided.")

    data_scaled = scaler.transform(data_matrix).astype(np.float32)
    return data_scaled, time_mark, scaler


def preprocess_airquality(
    xlsx_path: str,
    scale: bool = False,
    scaler: Optional[StandardScaler] = None,
):
    """
    UCI Air Quality (.xlsx)

    Steps:
    - read xlsx
    - merge Date + Time to datetime
    - replace -200 with NaN
    - fill NaN with column mean (only to make numeric preprocessing valid)
    - return raw values by default
    """
    df = pd.read_excel(xlsx_path)
    df = df.dropna(axis=1, how="all")

    df["date"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        errors="coerce",
        dayfirst=True,
    )

    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop(columns=["Date", "Time"])
    df = df.replace(-200, np.nan)

    cols_data = [c for c in df.columns if c != "date"]
    data_df = df[cols_data]

    if data_df.isna().any().any():
        data_df = data_df.fillna(data_df.mean())

    data_values = data_df.values.astype(np.float32)
    time_mark = build_time_mark_from_date(df["date"], freq="h")

    if not scale:
        return data_values, time_mark, None

    if scaler is None:
        raise ValueError("When scale=True in preprocess_airquality, scaler must be provided.")

    data_scaled = scaler.transform(data_values).astype(np.float32)
    return data_scaled, time_mark, scaler


# ---- Specific datasets ----

def preprocess_etth(path: str, scale: bool = False, scaler: Optional[StandardScaler] = None):
    return preprocess_csv(path, time_col="date", scale=scale, scaler=scaler)


def preprocess_ettm(path: str, scale: bool = False, scaler: Optional[StandardScaler] = None):
    return preprocess_csv(path, time_col="date", scale=scale, scaler=scaler)


def preprocess_ecl(path: str, scale: bool = False, scaler: Optional[StandardScaler] = None):
    return preprocess_csv(path, time_col="date", scale=scale, scaler=scaler)


def preprocess_weather(path: str, scale: bool = False, scaler: Optional[StandardScaler] = None):
    return preprocess_csv(path, time_col="date", scale=scale, scaler=scaler)


def preprocess_power(path: str, scale: bool = False, scaler: Optional[StandardScaler] = None):
    return preprocess_csv(path, time_col="date", scale=scale, scaler=scaler)


def preprocess_exchange(path: str, scale: bool = False, scaler: Optional[StandardScaler] = None):
    return preprocess_csv(path, time_col="date", scale=scale, scaler=scaler)


def preprocess_Solar(path: str = "", scale: bool = False, scaler: Optional[StandardScaler] = None):
    return preprocess_solar(scale=scale, scaler=scaler)


# ============================================================
# 6. Main Loader
#    核心修复：
#    1) 先读 raw data
#    2) 只在 train split 上 fit scaler
#    3) 再 transform 全部数据
#    4) 返回 scaler，供 test 阶段 inverse_transform 使用
# ============================================================

def load_forecasting_data(args):
    # --------------------------------------------------------
    # 1) Load raw data only (NO global fit_transform)
    # --------------------------------------------------------
    if args.data_type == "etth1":
        data_raw, _, _ = preprocess_etth("data/ETTh1.csv", scale=False)
    elif args.data_type == "etth2":
        data_raw, _, _ = preprocess_etth("data/ETTh2.csv", scale=False)
    elif args.data_type == "ettm1":
        data_raw, _, _ = preprocess_ettm("data/ETTm1.csv", scale=False)
    elif args.data_type == "ettm2":
        data_raw, _, _ = preprocess_ettm("data/ETTm2.csv", scale=False)
    elif args.data_type == "ecl":
        data_raw, _, _ = preprocess_ecl("data/ECL.csv", scale=False)
    elif args.data_type == "wth":
        data_raw, _, _ = preprocess_weather("data/WTH.csv", scale=False)
    elif args.data_type == "power":
        data_raw, _, _ = preprocess_power("data/Power.csv", scale=False)
    elif args.data_type == "exchange":
        data_raw, _, _ = preprocess_exchange("data/exchange.csv", scale=False)
    elif args.data_type == "solar":
        data_raw, _, _ = preprocess_Solar("data/solar.csv", scale=False)
    elif args.data_type == "air":
        data_raw, _, _ = preprocess_airquality(
            xlsx_path="data/AirQualityUCI.xlsx",
            scale=False,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.data_type}")

    data_raw = np.asarray(data_raw, dtype=np.float32)
    T, C = data_raw.shape

    # --------------------------------------------------------
    # 2) Split by time
    # --------------------------------------------------------
    train_T = int(T * (1 - args.val_ratio - args.test_ratio))
    val_T = int(T * args.val_ratio)

    train_range = (0, train_T)
    val_range = (train_T, train_T + val_T)
    test_range = (train_T + val_T, T)

    # --------------------------------------------------------
    # 3) Fit scaler ONLY on training range
    # --------------------------------------------------------
    scaler = StandardScaler()
    scaler.fit(data_raw[train_range[0]:train_range[1]])

    # transform all using train statistics
    data = scaler.transform(data_raw).astype(np.float32)

    # --------------------------------------------------------
    # 4) Build windows on normalized data
    # --------------------------------------------------------
    windows = build_forecasting_windows(
        data=data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.data_stride,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
    )

    train_ds = ForecastingWindowDataset(*windows["train"])
    val_ds = ForecastingWindowDataset(*windows["val"])
    test_ds = ForecastingWindowDataset(*windows["test"])

    if len(train_ds) == 0:
        raise ValueError(
            f"Train dataset is empty. Please check seq_len={args.seq_len}, "
            f"pred_len={args.pred_len}, data_stride={args.data_stride}, "
            f"and split sizes."
        )
    if len(val_ds) == 0:
        print("[WARN] Validation dataset is empty.")
    if len(test_ds) == 0:
        print("[WARN] Test dataset is empty.")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=forecasting_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=forecasting_collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=forecasting_collate,
    )

    return train_loader, val_loader, test_loader, C, scaler