import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model.get_model import get_model
from utils.best_name import build_best_model_name
from utils.data_factory import load_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate ChronoOrg for time-series forecasting.")

    data = parser.add_argument_group("Data")
    data.add_argument("--data_type", type=str, default="etth1", help="Dataset name")
    data.add_argument("--root_path", type=str, default="./data", help="Dataset root directory")
    data.add_argument("--data_path", type=str, default="ETTh1.csv", help="Dataset filename")
    data.add_argument("--freq", type=str, default="h", help="Frequency for time features")
    data.add_argument("--embed", type=str, default="timeF", help="Time-feature encoding")
    data.add_argument("--features", type=str, default="M", help="Feature mode: M, S, or MS")
    data.add_argument("--target", type=str, default="OT", help="Target column for univariate forecasting")
    data.add_argument("--scale", type=int, default=1, help="Standardize data (1=True, 0=False)")
    data.add_argument("--seq_len", type=int, default=96, help="Input sequence length")
    data.add_argument("--label_len", type=int, default=48, help="History length included in dataset targets")
    data.add_argument("--pred_len", type=int, default=96, help="Forecast horizon")
    data.add_argument("--data_stride", type=int, default=1, help="Legacy stride label used in checkpoint filenames")

    model = parser.add_argument_group("ChronoOrg")
    model.add_argument("--model_name", type=str.lower, default="chronoorg", choices=["chronoorg"])
    model.add_argument("--patch_len", type=int, default=16, help="Temporal tokenizer patch length")
    model.add_argument("--patch_stride", type=int, default=8, help="Temporal tokenizer patch stride")
    model.add_argument("--tokenizer_bias", type=int, default=1, help="Use tokenizer bias (1=True, 0=False)")
    model.add_argument("--d_model", type=int, default=64, help="Initial stage dimension")
    model.add_argument(
        "--stage_dims", type=int, nargs="+", default=None,
        help="Stage dimensions; defaults to progressive doubling from d_model",
    )
    model.add_argument("--depths", type=int, nargs="+", default=[2, 2, 2], help="Blocks per stage")
    model.add_argument("--num_heads", type=int, nargs="+", default=[2, 4, 8], help="Attention heads per stage")
    model.add_argument("--window_size", type=int, nargs="+", default=[8, 8, 4], help="Window size per stage")
    model.add_argument(
        "--global_tokens", type=int, nargs="+", default=None,
        help="SCE output tokens per stage; defaults to twice each window_size",
    )
    model.add_argument(
        "--stage_use_global", type=str, nargs="+", default=None,
        help="Per-block CCTA flags for each stage (0=BTI, 1=CCTA), e.g. 0,1 0,1 0,1,0,1",
    )
    model.add_argument("--mlp_ratio", type=float, default=4.0, help="FFN expansion ratio")
    model.add_argument(
        "--ffn_hidden_dim", type=int, nargs="+", default=None,
        help="FFN hidden dimensions per stage; defaults to mlp_ratio times stage dimensions",
    )
    model.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    model.add_argument("--qkv_bias", type=int, default=1, help="Use QKV projection bias (1=True, 0=False)")
    model.add_argument("--layer_scale", type=float, default=None, help="Optional LayerScale initial value")
    model.add_argument("--downsample", type=int, default=1, help="Use Temporal Merge between stages (1=True, 0=False)")
    model.add_argument("--period", type=int, default=24, help="TI period in raw time steps")
    model.add_argument("--pooling", type=str, default="mean", choices=["mean", "last"], help="Prediction-head pooling")
    model.add_argument("--revin", type=int, default=1, help="Use RevIN (1=True, 0=False)")
    model.add_argument("--revin_eps", type=float, default=1e-5, help="RevIN numerical stability constant")
    model.add_argument("--affine", type=int, default=1, help="Use RevIN affine parameters (1=True, 0=False)")
    model.add_argument(
        "--subtract_last", type=int, default=0,
        help="Center RevIN on the last observation (1=True, 0=False)",
    )

    training = parser.add_argument_group("Training")
    training.add_argument("--batch_size", type=int, default=16)
    training.add_argument("--lr", type=float, default=0.0001, help="Adam learning rate")
    training.add_argument("--epochs", type=int, default=100)
    training.add_argument("--patience", type=int, default=5, help="Early-stopping patience")
    training.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--save_dir", type=str, default="result_forecast", help="Checkpoint directory")
    return parser.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(preds: np.ndarray, gts: np.ndarray):
    """Compute metrics for (N, H, C) arrays on their supplied scale."""
    eps = 1e-8

    diff = preds - gts
    mse = np.mean(diff ** 2)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(mse)

    numerator = np.abs(preds - gts)
    denominator = (np.abs(gts) + np.abs(preds)) / 2.0
    smape = np.mean(numerator / (denominator + eps))

    wape = np.sum(np.abs(preds - gts)) / (np.sum(np.abs(gts)) + eps)

    rmse_global = np.sqrt(np.mean(diff ** 2))
    scale_true = np.sqrt(np.mean(gts ** 2))
    scale_pred = np.sqrt(np.mean(preds ** 2))
    tic = rmse_global / (scale_true + scale_pred + eps)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "smape": smape,
        "wape": wape,
        "tic": tic,
    }


def align_to_pred_len(tensor, pred_len):
    """Keep the final pred_len steps of a [B, T, ...] tensor."""
    if tensor is None:
        return None
    return tensor[:, -pred_len:, ...]


def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    train_loader, val_loader, test_loader, input_dim, _ = load_data(args)

    print(f"[INFO] Dataset={args.data_type}, input_dim={input_dim}")
    print(f"[INFO] Device={device}")

    model = get_model(args, input_dim=input_dim).to(device)
    print("Model device:", next(model.parameters()).device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_model_path = os.path.join(args.save_dir, build_best_model_name(args))
    best_val_loss = float("inf")
    early_stop_cnt = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for idx, batch in enumerate(train_loader):
            x, y, _, _, _ = batch
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            output = model(x)

            output = align_to_pred_len(output, args.pred_len)
            y = align_to_pred_len(y, args.pred_len)

            if output.shape != y.shape:
                raise ValueError(
                    f"[TRAIN] Model output shape {output.shape} does not match target shape {y.shape}"
                )

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if idx % 50 == 0:
                print(f"Epoch {epoch + 1} | idx {idx} | Train MSE {loss.item():.6f}")

        epoch_loss /= max(len(train_loader), 1)
        print(f"[TRAIN] Epoch {epoch + 1}, MSE={epoch_loss:.6f}, time={time.time() - t0:.1f}s")

        model.eval()

        val_loss = 0.0
        val_start_time = time.time()

        with torch.no_grad():
            for batch in val_loader:
                x, y, _, _, _ = batch
                x = x.to(device)
                y = y.to(device)

                output = model(x)

                output = align_to_pred_len(output, args.pred_len)
                y = align_to_pred_len(y, args.pred_len)

                if output.shape != y.shape:
                    raise ValueError(
                        f"[VAL] Model output shape {output.shape} does not match target shape {y.shape}"
                    )

                loss = criterion(output, y)
                val_loss += loss.item()

        val_time = time.time() - val_start_time
        val_loss /= max(len(val_loader), 1)

        print(f"[VAL] Epoch {epoch + 1}, MSE={val_loss:.6f}, Time={val_time:.4f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_cnt = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] Best model saved to: {best_model_path}")
        else:
            early_stop_cnt += 1
            print(f"[INFO] Early stop counter: {early_stop_cnt}/{args.patience}")
            if early_stop_cnt >= args.patience:
                print(f"[Early Stop] triggered at epoch {epoch + 1}")
                break

    print("[INFO] Loading best model for test...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds = []
    gts = []

    with torch.no_grad():
        for batch in test_loader:
            x, y, _, _, _ = batch
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            output = align_to_pred_len(output, args.pred_len)
            y = align_to_pred_len(y, args.pred_len)

            if output.shape != y.shape:
                raise ValueError(
                    f"[TEST] Model output shape {output.shape} does not match target shape {y.shape}"
                )

            preds.append(output.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)

    metrics = compute_metrics(preds, gts)

    print(
        f"[TEST] "
        f"Model={args.model_name} | "
        f"Data={args.data_type} | "
        f"Prediction={args.pred_len} | "
        f"MSE={metrics['mse']:.4f} | "
        f"MAE={metrics['mae']:.4f} | "
        f"RMSE={metrics['rmse']:.4f}"
    )


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args = parse_args()
    train(args)
