from .data_loader import *
from torch.utils.data import DataLoader


data_dict = {
    # =========================
    # ETT
    # =========================
    'etth1': Dataset_ETT_hour,
    'etth2': Dataset_ETT_hour,
    'ettm1': Dataset_ETT_minute,
    'ettm2': Dataset_ETT_minute,

    # =========================
    # TXT / 常规多变量
    # =========================
    'solar': Dataset_Solar,
    'ecl': Dataset_TXT,        # electricity.txt / csv
    'exchange': Dataset_TXT,
    'traffic': Dataset_TXT,
    'custom': Dataset_Custom,
    'air': Dataset_Air,

    # =========================
    # 图时序（NPZ）
    # =========================
    'pems03': Dataset_PEMS,
    'pems04': Dataset_PEMS,
    'pems07': Dataset_PEMS,
    'pems08': Dataset_PEMS,
}

DATA_PATH_DICT = {
    "etth1": ("data", "ETTh1.csv"),
    "etth2": ("data", "ETTh2.csv"),
    "ettm1": ("data", "ETTm1.csv"),
    "ettm2": ("data", "ETTm2.csv"),

    "solar": ("data", "solar_AL.txt"),
    "ecl": ("data", "electricity.txt"),
    "exchange": ("data", "exchange_rate.txt"),
    "traffic": ("data", "traffic.txt"),
    "air": ("data", "AirQualityUCI.xlsx"),

    "pems03": ("data", "PEMS03.npz"),
    "pems04": ("data", "pems04.npz"),
    "pems07": ("data", "PEMS07.npz"),
    "pems08": ("data", "pems08.npz"),
}


def load_data(args):

    Data = data_dict[args.data_type]
    args.root_path, args.data_path = DATA_PATH_DICT[args.data_type]
    timeenc = 0 if args.embed != 'timeF' else 1

    def build(flag, shuffle):
        dataset = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=timeenc,
            freq=args.freq
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            drop_last=True
        )

        return dataset, loader

    train_set, train_loader = build('train', True)
    val_set, val_loader = build('val', False)
    test_set, test_loader = build('test', False)

    C = train_set.get_data_dim()
    scaler = train_set.get_scaler()

    print(f"[INFO] train={len(train_set)} val={len(val_set)} test={len(test_set)}")
    print(f"[INFO] Dataset={args.data_type}, input_dim={C}")

    return train_loader, val_loader, test_loader, C, scaler