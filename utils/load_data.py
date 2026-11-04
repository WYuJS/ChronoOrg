import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features


# ============================================================
# 1. Dataset
# ============================================================

class ForecastingDatasetUniform(Dataset):
    def __init__(self, data, data_stamp, seq_len, pred_len, start, end):
        self.data = data
        self.data_stamp = data_stamp
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.start = start
        self.end = end

    def __len__(self):
        return self.end - self.start - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        index = index + self.start

        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        x = self.data[s_begin:s_end]
        y = self.data[r_begin:r_end]

        x_mark = self.data_stamp[s_begin:s_end]
        y_mark = self.data_stamp[r_begin:r_end]

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(x_mark).float(),
            torch.from_numpy(y_mark).float()
        )

# ============================================================
# 2. Collate
# ============================================================

def forecasting_collate(batch):
    xs, ys, x_marks, y_marks = zip(*batch)

    B = len(xs)
    C = xs[0].shape[1]
    H = ys[0].shape[0]
    Tm = x_marks[0].shape[1]

    lengths = torch.tensor([x.shape[0] for x in xs])
    L_max = lengths.max().item()

    x_batch = torch.zeros(B, L_max, C)
    y_batch = torch.zeros(B, H, C)

    x_mark_batch = torch.zeros(B, L_max, Tm)
    y_mark_batch = torch.zeros(B, H, Tm)

    for i, (x, y, xm, ym) in enumerate(zip(xs, ys, x_marks, y_marks)):
        x_batch[i, :x.shape[0]] = x
        y_batch[i] = y
        x_mark_batch[i, :xm.shape[0]] = xm
        y_mark_batch[i] = ym

    return x_batch, y_batch, x_mark_batch, y_mark_batch, lengths


# ============================================================
# 3. Sliding Window
# ============================================================

def build_forecasting_windows(
    data,
    data_stamp,
    seq_len,
    pred_len,
    stride,
    train_range,
    val_range,
    test_range,
):
    splits = {
        "train": train_range,
        "val": val_range,
        "test": test_range,
    }

    results = {}

    for split, (start, end) in splits.items():

        max_t = end - seq_len - pred_len + 1
        if max_t <= start:
            results[split] = ([], [], [], [])
            continue

        cur_stride = stride if split == "train" else 1

        x_list, y_list = [], []
        x_mark_list, y_mark_list = [], []

        for t in range(start, max_t, cur_stride):

            s_begin = t
            s_end = t + seq_len

            r_begin = s_end
            r_end = r_begin + pred_len

            x = data[s_begin:s_end]
            y = data[r_begin:r_end]

            x_mark = data_stamp[s_begin:s_end]
            y_mark = data_stamp[r_begin:r_end]

            x_list.append(x.astype(np.float32))
            y_list.append(y.astype(np.float32))
            x_mark_list.append(x_mark.astype(np.float32))
            y_mark_list.append(y_mark.astype(np.float32))

        results[split] = (x_list, y_list, x_mark_list, y_mark_list)

    return results


# ============================================================
# 4. NEW: TXT / NPZ loader（核心改动）
# ============================================================


def preprocess_csv(path, timeenc=1, freq='h'):
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # -------- 时间特征 --------
    df_stamp = df[["date"]].copy()
    data_stamp = time_features(df_stamp, timeenc=timeenc, freq=freq)

    # -------- 数值 --------
    data = df.drop(columns=["date"]).values.astype(np.float32)

    return data, data_stamp




def preprocess_txt(path):
    data = np.loadtxt(path, delimiter=',').astype(np.float32)
    data_stamp = np.zeros((data.shape[0], 1), dtype=np.float32)
    return data, data_stamp


def preprocess_npz(path):
    data = np.load(path)["data"].astype(np.float32)

    # PEMS: (T, N, F) -> 只取flow
    if data.ndim == 3:
        data = data[:, :, 0]

    data_stamp = np.zeros((data.shape[0], 1), dtype=np.float32)
    return data, data_stamp


# ============================================================
# 5. Main Loader（核心改动）
# ============================================================

def load_forecasting_data(args):

    # ------------------------
    # CSV datasets
    # ------------------------
    if args.data_type == "etth1":
        data_raw, data_stamp = preprocess_csv("data/ETTh1.csv", args.timeenc, args.freq)
    
    elif args.data_type == "etth2":
        data_raw, data_stamp = preprocess_csv("data/ETTh2.csv", args.timeenc, args.freq)
    
    elif args.data_type == "ettm1":
        data_raw, data_stamp = preprocess_csv("data/ETTm1.csv", args.timeenc, args.freq)
    
    elif args.data_type == "ettm2":
        data_raw, data_stamp = preprocess_csv("data/ETTm2.csv", args.timeenc, args.freq)

    # ------------------------
    # TXT datasets
    # ------------------------
    elif args.data_type == "solar":
        data_raw, data_stamp = preprocess_txt("data/solar_AL.txt")

    elif args.data_type == "ecl":
        data_raw, data_stamp = preprocess_txt("data/electricity.txt")

    elif args.data_type == "exchange":
        data_raw, data_stamp = preprocess_txt("data/exchange_rate.txt")

    elif args.data_type == "traffic":
        data_raw, data_stamp = preprocess_txt("data/traffic.txt")

    # ------------------------
    # PEMS datasets（NPZ）
    # ------------------------
    elif args.data_type == "pems03":
        data_raw, data_stamp = preprocess_npz("data/PEMS03.npz")

    elif args.data_type == "pems04":
        data_raw, data_stamp = preprocess_npz("data/PEMS04.npz")

    elif args.data_type == "pems07":
        data_raw, data_stamp = preprocess_npz("data/PEMS07.npz")

    elif args.data_type == "pems08":
        data_raw, data_stamp = preprocess_npz("data/PEMS08.npz")

    else:
        raise ValueError(f"Unknown dataset: {args.data_type}")

    data_raw = np.asarray(data_raw, dtype=np.float32)
    T, C = data_raw.shape

    # ------------------------
    # split
    # ------------------------
    if args.data_type in ["etth1", "etth2"]:
        train_len = 12 * 30 * 24
        val_len = 4 * 30 * 24
        test_len = 4 * 30 * 24

    elif args.data_type in ["ettm1", "ettm2"]:
        scale = 4
        train_len = 12 * 30 * 24 * scale
        val_len = 4 * 30 * 24 * scale
        test_len = 4 * 30 * 24 * scale

    else:
        train_len = int(T * (1 - args.val_ratio - args.test_ratio))
        val_len = int(T * args.val_ratio)
        test_len = T - train_len - val_len

    train_range = (0, train_len)

    val_range = (
        train_len - args.seq_len,
        train_len + val_len
    )

    test_range = (
        train_len + val_len - args.seq_len,
        min(train_len + val_len + test_len, T)
    )

    # ------------------------
    # scaler
    # ------------------------
    scaler = StandardScaler()
    scaler.fit(data_raw[:train_len])
    data = scaler.transform(data_raw).astype(np.float32)

    # ------------------------
    # windows
    # ------------------------
    windows = build_forecasting_windows(
        data=data,
        data_stamp=data_stamp,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.data_stride,
        train_range=train_range,
        val_range=val_range,
        test_range=test_range,
    )

    # -------- dataset --------
    train_ds = ForecastingDatasetUniform(
        data, data_stamp,
        args.seq_len, args.pred_len,
        *train_range
    )

    val_ds = ForecastingDatasetUniform(
        data, data_stamp,
        args.seq_len, args.pred_len,
        *val_range
    )

    test_ds = ForecastingDatasetUniform(
        data, data_stamp,
        args.seq_len, args.pred_len,
        *test_range
    )

    print(f"[INFO] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        # shuffle=False,  # 注意：不打乱，保持时间顺序
        shuffle=True,  # 训练时打乱，验证和测试时不打乱
        # drop_last=True,
        num_workers=args.num_workers,
        # collate_fn=forecasting_collate,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # collate_fn=forecasting_collate,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        # collate_fn=forecasting_collate,
    )

    return train_loader, val_loader, test_loader, C, scaler