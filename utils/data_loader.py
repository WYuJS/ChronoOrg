import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features


# =========================================================
# Base Dataset（统一 + TQNet + 你现在）
# =========================================================
class BaseDataset(Dataset):
    def __init__(
        self,
        root_path,
        data_path,
        flag,
        size,
        features,
        target,
        scale,
        timeenc,
        freq,
        cycle=24,
    ):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.cycle = cycle

        self.root_path = root_path
        self.data_path = data_path

        self.__read_data__()

    # =====================================================
    # 时间编码
    # =====================================================
    def _time_features(self, df_stamp):
        df_stamp = df_stamp.copy()

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.dt.month
            df_stamp['day'] = df_stamp.date.dt.day
            df_stamp['weekday'] = df_stamp.date.dt.weekday
            df_stamp['hour'] = df_stamp.date.dt.hour
            return df_stamp.drop(['date'], axis=1).values
        else:
            return time_features(df_stamp, timeenc=1, freq=self.freq)

    # =====================================================
    # window
    # =====================================================
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        x = self.data_x[s_begin:s_end]
        y = self.data_y[r_begin:r_end]
        xm = self.data_stamp[s_begin:s_end]
        ym = self.data_stamp[r_begin:r_end]

        cycle = torch.tensor(self.cycle_index[s_end])

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(xm).float(),
            torch.from_numpy(ym).float(),
            cycle
        )

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def get_data_dim(self):
        return self.data_x.shape[1]

    def get_scaler(self):
        return self.scaler


# =========================================================
# ETTh
# =========================================================
class Dataset_ETT_hour(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_stamp = df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.features == 'S':
            df_data = df[[self.target]]
        else:
            df_data = df.iloc[:, 1:]

        data = df_data.values

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = self._time_features(df_stamp.iloc[border1:border2].copy())

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# ETTm
# =========================================================
class Dataset_ETT_minute(Dataset_ETT_hour):
    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        scale = 4
        border1s = [
            0,
            12 * 30 * 24 * scale - self.seq_len,
            12 * 30 * 24 * scale + 4 * 30 * 24 * scale - self.seq_len
        ]
        border2s = [
            12 * 30 * 24 * scale,
            12 * 30 * 24 * scale + 4 * 30 * 24 * scale,
            12 * 30 * 24 * scale + 8 * 30 * 24 * scale
        ]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_stamp = df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.features == 'S':
            df_data = df[[self.target]]
        else:
            df_data = df.iloc[:, 1:]

        data = df_data.values

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = self._time_features(df_stamp.iloc[border1:border2].copy())

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# Custom / ECL / Traffic / Exchange（统一）
# =========================================================
class Dataset_Custom(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(os.path.join(self.root_path, self.data_path))

        num_train = int(len(df) * 0.7)
        num_val = int(len(df) * 0.1)
        num_test = len(df) - num_train - num_val

        border1s = [0, num_train - self.seq_len, len(df) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df)]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_stamp = df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.features == 'S':
            df_data = df[[self.target]]
        else:
            df_data = df.drop(columns=['date'])

        data = df_data.values

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = self._time_features(df_stamp.iloc[border1:border2].copy())

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# AirQualityUCI（支持 xlsx / csv）
# =========================================================
class Dataset_Air(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()
        file_path = os.path.join(self.root_path, self.data_path)

        if self.data_path.lower().endswith('.xlsx') or self.data_path.lower().endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        # -------------------------------------------------
        # 去掉全空列 / 常见无效列
        # -------------------------------------------------
        df = df.dropna(axis=1, how='all')

        unnamed_cols = [c for c in df.columns if str(c).lower().startswith('unnamed')]
        if len(unnamed_cols) > 0:
            df = df.drop(columns=unnamed_cols)

        # -------------------------------------------------
        # 构造 date 列
        # 兼容：
        # 1) 已有 date 列
        # 2) 有 Date 和 Time 两列
        # -------------------------------------------------
        cols_lower = {str(c).lower(): c for c in df.columns}

        if 'date' in cols_lower:
            date_col = cols_lower['date']
            df['date'] = pd.to_datetime(df[date_col], errors='coerce')
        elif 'datetime' in cols_lower:
            dt_col = cols_lower['datetime']
            df['date'] = pd.to_datetime(df[dt_col], errors='coerce')
        elif 'date' in df.columns and 'time' in df.columns:
            df['date'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce'
            )
        elif 'Date' in df.columns and 'Time' in df.columns:
            df['date'] = pd.to_datetime(
                df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                errors='coerce'
            )
        else:
            # 如果完全没有时间列，就人工构造
            df['date'] = pd.date_range(start='2000-01-01', periods=len(df), freq=self.freq)

        # -------------------------------------------------
        # 删除原始时间字符串列，避免进入特征
        # -------------------------------------------------
        drop_cols = []
        for c in df.columns:
            if c == 'date':
                continue
            if str(c).lower() in ['date', 'time', 'datetime']:
                drop_cols.append(c)
        if len(drop_cols) > 0:
            df = df.drop(columns=drop_cols)

        # -------------------------------------------------
        # 缺失值处理：AirQualityUCI 里常见 -200 表示缺失
        # -------------------------------------------------
        df = df.replace(-200, np.nan)
        df = df.replace(-200.0, np.nan)

        # 仅保留数值列 + date
        feature_cols = []
        for c in df.columns:
            if c == 'date':
                continue
            df[c] = pd.to_numeric(df[c], errors='coerce')
            feature_cols.append(c)

        df = df[['date'] + feature_cols]

        # 按时间排序
        df = df.sort_values('date').reset_index(drop=True)

        # 去掉无法解析时间的行
        df = df.dropna(subset=['date']).reset_index(drop=True)

        # 先做一个简单插值，再前后填充
        for c in feature_cols:
            df[c] = df[c].interpolate(method='linear', limit_direction='both')
            df[c] = df[c].ffill().bfill()

        # 如果还有全空列，删掉
        valid_feature_cols = [c for c in feature_cols if not df[c].isna().all()]
        df = df[['date'] + valid_feature_cols]

        if len(valid_feature_cols) == 0:
            raise ValueError("Air dataset has no valid numeric feature columns after preprocessing.")

        num_train = int(len(df) * 0.7)
        num_val = int(len(df) * 0.1)
        num_test = len(df) - num_train - num_val

        border1s = [0, num_train - self.seq_len, len(df) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(df)]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        df_stamp = df[['date']].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        if self.features == 'S':
            if self.target not in df.columns:
                raise ValueError(f"Target column '{self.target}' not found in air dataset.")
            df_data = df[[self.target]]
        else:
            df_data = df.drop(columns=['date'])

        data = df_data.values.astype(np.float32)

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = self._time_features(df_stamp.iloc[border1:border2].copy())

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


class Dataset_TXT(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()
        data = np.loadtxt(os.path.join(self.root_path, self.data_path), delimiter=',')

        T = len(data)
        num_test = int(T * 0.2)
        num_val = int(T * 0.1)
        num_train = T - num_val - num_test

        border1s = [0, num_train - self.seq_len, T - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, T]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = np.zeros((border2 - border1, 1), dtype=np.float32)

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# Solar / TXT
# =========================================================
class Dataset_Solar(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()

        data = np.loadtxt(os.path.join(self.root_path, self.data_path), delimiter=',')

        num_train = int(len(data) * 0.7)
        num_val = int(len(data) * 0.1)
        num_test = len(data) - num_train - num_val

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(data)]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = np.zeros((len(self.data_x), 1), dtype=np.float32)

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]


# =========================================================
# PEMS
# =========================================================
class Dataset_PEMS(BaseDataset):
    def __read_data__(self):
        self.scaler = StandardScaler()

        data = np.load(os.path.join(self.root_path, self.data_path))['data']

        if data.ndim == 3:
            data = data[:, :, 0]

        num_train = int(len(data) * 0.6)
        num_val = int(len(data) * 0.2)
        num_test = len(data) - num_train - num_val

        border1s = [0, num_train - self.seq_len, len(data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_val, len(data)]

        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        if self.scale:
            self.scaler.fit(data[border1s[0]:border2s[0]])
            data = self.scaler.transform(data)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = np.zeros((len(self.data_x), 1), dtype=np.float32)

        self.cycle_index = (np.arange(len(data)) % self.cycle)[border1:border2]