import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
import argparse
import os

# from utils.load_data import load_forecasting_data
from utils.data_factory import load_data
from utils.timefeatures import infer_time_dim
from utils.chronoorg_configs import process_chronoorg_args
from utils.best_name import build_best_model_name
from model.get_model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ChronoOrg Time-Series Forecasting")

    # -----------------------------
    # Dataset / Experiment Settings
    # -----------------------------
    parser.add_argument('--data_type', type=str, default='etth1',
                        help='Dataset identifier (e.g., etth1, etth2, weather, wind)')
    parser.add_argument('--root_path', type=str, default='./data', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--time_col', type=str, default='date',
                        help='Name of the timestamp column')
    parser.add_argument('--model_name', type=str, default='chronoorg',
                        help='Model architecture to use')
    parser.add_argument('--timeenc', type=int, default=1,
                        help='time encoding type (0: discrete, 1: continuous)')
    parser.add_argument('--freq', type=str, default='h',
                        help='time frequency (h, t, s, etc.)')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--scale', type=int, default=1)

    # -----------------------------
    # Hyperparameters
    # -----------------------------
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Input sequence length (lookback window size)')
    parser.add_argument('--label_len', type=int, default=48,
                        help='Start token length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction horizon length')
    parser.add_argument('--patch_len', type=int, default=16,
                        help='Patch length')
    parser.add_argument('--patch_stride', type=int, default=8,
                        help='Stride for patch construction')
    parser.add_argument('--data_stride', type=int, default=1,
                        help='Sliding window stride for dataset sample generation')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test split ratio')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Mini-batch size for training')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # -----------------------------
    # Model-specific Parameters
    # -----------------------------
    parser.add_argument('--d_model', type=int, default=64,
                        help='Transformer model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='Feed-forward network hidden dimension')
    parser.add_argument('--enc_layers', type=int, default=3,
                        help='Number of encoder layers')
    parser.add_argument('--dec_layers', type=int, default=1,
                        help='Number of decoder layers')
    parser.add_argument('--use_gating', type=int, default=1,
                        help='Enable gated attention mechanism (1=True, 0=False)')
    parser.add_argument('--tau', type=float, default=24.0,
                        help='Temporal distance scaling factor in Dt')
    parser.add_argument('--lambda_rel', type=float, default=0.5,
                        help='Reliability decay strength')
    parser.add_argument('--d_state', type=int, default=64,
                        help='Latent state embedding dimension for Ds')
    parser.add_argument('--state_hidden', type=int, default=128,
                        help='Hidden dimension of state embedding MLP')
    parser.add_argument('--pair_hidden', type=int, default=128,
                        help='Hidden dimension of pairwise distance MLP')

    # -----------------------------
    # PatchTST
    # -----------------------------
    parser.add_argument('--fc_dropout', type=float, default=0.05,
                        help='Fully connected dropout in PatchTST head')
    parser.add_argument('--head_dropout', type=float, default=0.0,
                        help='Dropout rate for prediction head')
    parser.add_argument('--padding_patch', type=str, default='end',
                        help='Patch padding mode: None or end')
    parser.add_argument('--revin', type=int, default=1,
                        help='Use RevIN normalization (1=True, 0=False)')
    parser.add_argument('--affine', type=int, default=1,
                        help='Use affine transformation in RevIN (1=True, 0=False)')
    parser.add_argument('--subtract_last', type=int, default=0,
                        help='RevIN normalization mode: 0=subtract mean, 1=subtract last')
    parser.add_argument('--decomposition', type=int, default=0,
                        help='Use decomposition branch (1=True, 0=False)')
    parser.add_argument('--kernel_size', type=int, default=25,
                        help='Kernel size for series decomposition')
    parser.add_argument('--individual', type=int, default=0,
                        help='Use individual prediction head for each channel (1=True, 0=False)')

    # -----------------------------
    # DeformableTST
    # -----------------------------
    parser.add_argument('--n_vars', type=int, default=7,
                        help='Number of variables in the input series (kept for compatibility; current get_model uses input_dim instead)')
    parser.add_argument('--revin_affine', type=int, default=0,
                        help='Use affine transformation in RevIN (1=True, 0=False)')
    parser.add_argument('--revin_subtract_last', type=int, default=0,
                        help='RevIN normalization mode: 0=subtract mean, 1=subtract last value')

    parser.add_argument('--stem_ratio', type=int, default=1,
                        help='Down-sampling ratio in the stem layer')
    parser.add_argument('--down_ratio', type=int, default=2,
                        help='Down-sampling ratio between adjacent stages')
    parser.add_argument('--fmap_size', type=int, default=96,
                        help='Feature map length of the input series (kept for compatibility; current get_model uses seq_len instead)')

    parser.add_argument('--dims', nargs='+', type=int, default=[16, 32, 64, 128],
                        help='Hidden dimensions for each stage')
    parser.add_argument('--d_depths', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='Number of Transformer blocks in each stage')
    parser.add_argument('--drop_path_rate', type=float, default=0.3,
                        help='Drop path rate')
    parser.add_argument('--layer_scale_value', nargs='+', type=float, default=[-1, -1, -1, -1],
                        help='Layer scale initialization values for each stage')

    parser.add_argument('--use_pe', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='Use positional encoding in each stage (1=True, 0=False)')
    parser.add_argument('--use_lpu', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='Use Local Perception Unit in each stage (1=True, 0=False)')
    parser.add_argument('--local_kernel_size', nargs='+', type=int, default=[3, 3, 3, 3],
                        help='Kernel size of Local Perception Unit in each stage')

    parser.add_argument('--expansion', type=int, default=4,
                        help='Expansion ratio of FFN')
    parser.add_argument('--use_dwc_mlp', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='Use depthwise-convolutional MLP in each stage (1=True, 0=False)')

    parser.add_argument('--heads', nargs='+', type=int, default=[4, 8, 16, 32],
                        help='Number of attention heads in each stage')
    parser.add_argument('--attn_drop', type=float, default=0.0,
                        help='Dropout rate for attention map')
    parser.add_argument('--proj_drop', type=float, default=0.0,
                        help='Dropout rate for attention projection')

    parser.add_argument('--stage_spec', nargs='+', type=str, default=['D', 'D', 'D', 'D'],
                        help='Block type specification for each stage')
    parser.add_argument('--d_window_size', nargs='+', type=int, default=[8, 8, 8, 8],
                        help='Window size for window attention in each stage')
    parser.add_argument('--nat_ksize', nargs='+', type=int, default=[3, 3, 3, 3],
                        help='Kernel size for neighborhood attention in each stage')

    parser.add_argument('--offset_kernel_size', nargs='+', type=int, default=[9, 7, 5, 3],
                        help='Kernel size of the offset sub-network in each stage')
    parser.add_argument('--offset_stride', nargs='+', type=int, default=[8, 4, 2, 1],
                        help='Stride of the offset sub-network in each stage')
    parser.add_argument('--n_groups', nargs='+', type=int, default=[2, 4, 8, 16],
                        help='Number of offset groups in each stage')
    parser.add_argument('--offset_range_factor', nargs='+', type=float, default=[-1, -1, -1, -1],
                        help='Range scaling factor for learned offsets in each stage')
    parser.add_argument('--no_off', nargs='+', type=int, default=[0, 0, 0, 0],
                        help='Disable learned offsets in each stage (1=True, 0=False)')
    parser.add_argument('--dwc_pe', nargs='+', type=int, default=[0, 0, 0, 0],
                        help='Use depthwise-convolutional positional encoding in each stage (1=True, 0=False)')
    parser.add_argument('--fixed_pe', nargs='+', type=int, default=[0, 0, 0, 0],
                        help='Use fixed positional encoding in each stage (1=True, 0=False)')
    parser.add_argument('--log_cpb', nargs='+', type=int, default=[0, 0, 0, 0],
                        help='Use Swin-v2 style log continuous position bias in each stage (1=True, 0=False)')
    parser.add_argument('--head_type', type=str, default='Flatten',
                        help='Prediction head type')
    parser.add_argument('--use_head_norm', type=int, default=1,
                        help='Use final layer normalization before prediction head (1=True, 0=False)')

    # -----------------------------
    # CATS
    # -----------------------------
    parser.add_argument('--QAM_start', type=float, default=0.1,
                        help='masking start probability')
    parser.add_argument('--QAM_end', type=float, default=0.3,
                        help='masking end probability')
    parser.add_argument('--pct_start', type=float, default=0.3,
                        help='pct_start')
    parser.add_argument('--query_independence', action='store_true', default=False,
                        help='sharing query across dimension')
    parser.add_argument('--store_attn', action='store_true', default=False,
                        help='store attention score')

    # -----------------------------
    # TQNet
    # -----------------------------
    parser.add_argument('--cycle', type=int, default=24,
                        help='Cycle length used in TQNet')
    parser.add_argument('--model_type', type=str, default='mlp',
                        help='TQNet backbone type, options: [linear, mlp]')
    parser.add_argument('--use_revin', type=int, default=1,
                        help='Use RevIN in TQNet (1=True, 0=False)')

    # -----------------------------
    # xPatch
    # -----------------------------
    parser.add_argument('--ma_type', type=str, default='ema',
                        help='Moving-average type in xPatch, options: [reg, ema, dema]')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Alpha coefficient for moving-average module in xPatch')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='Beta coefficient for moving-average module in xPatch')

    # -----------------------------
    # TSMixer
    # -----------------------------
    parser.add_argument('--n_block', type=int, default=2,
                        help='Number of mixer blocks used in TSMixer')
    parser.add_argument('--norm_type', type=str, default='B',
                        choices=['B', 'L'],
                        help='Normalization type used in TSMixer, options: [B, L]')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu'],
                        help='Activation function used in TSMixer')

    # -----------------------------
    # ParallelTime
    # -----------------------------
    parser.add_argument("--head_size", type=int, default=4)
    parser.add_argument("--ffn_multiplier", type=float, default=3.0)
    parser.add_argument("--patches_window_len", type=int, default=4)
    parser.add_argument("--num_register_tokens", type=int, default=32)
    parser.add_argument("--d_conv", type=int, default=2)
    parser.add_argument("--expend_ratio_scaler", type=int, default=16)
    parser.add_argument("--proj_expend_ratio", type=int, default=4)
    parser.add_argument("--proj_squeeze_ratio", type=int, default=2)
    parser.add_argument("--dropout_proj", type=float, default=0.05)


    # ============================================================
    # ModernTCN parser
    # ============================================================

    parser.add_argument('--mtcn_stem_ratio', type=int, default=6)
    parser.add_argument('--mtcn_downsample_ratio', type=int, default=2)
    parser.add_argument('--mtcn_ffn_ratio', type=int, default=2)

    parser.add_argument('--mtcn_num_blocks', type=int, nargs='+', default=[1, 1, 1, 1])
    parser.add_argument('--mtcn_large_size', type=int, nargs='+', default=[31, 29, 27, 13])
    parser.add_argument('--mtcn_small_size', type=int, nargs='+', default=[5, 5, 5, 5])

    parser.add_argument('--mtcn_dims', type=int, nargs='+', default=[64, 64, 64, 64])
    parser.add_argument('--mtcn_dw_dims', type=int, nargs='+', default=[64, 64, 64, 64])

    parser.add_argument('--mtcn_small_kernel_merged', type=int, default=0)
    parser.add_argument('--mtcn_use_multi_scale', type=int, default=0)  


    # -----------------------------
    # SST
    # -----------------------------
    parser.add_argument('--m_layers', type=int, default=3,
                        help='Number of Mamba layers in SST long-range branch')
    parser.add_argument('--m_patch_len', type=int, default=48,
                        help='Patch length of SST Mamba long-range branch')
    parser.add_argument('--m_stride', type=int, default=24,
                        help='Patch stride of SST Mamba long-range branch')
    parser.add_argument('--local_ws', type=int, default=3,
                        help='Local window size of SST short-range transformer branch; must be odd')
    parser.add_argument('--concat', type=int, default=1,
                        help='Fusion mode in SST: 1=concat, 0=add')

    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='Maximum sequence length for SST positional encoding')
    parser.add_argument('--d_k', type=int, default=None,
                        help='Key dimension in SST attention; None means d_model // n_heads')
    parser.add_argument('--d_v', type=int, default=None,
                        help='Value dimension in SST attention; None means d_model // n_heads')
    parser.add_argument('--norm', type=str, default='BatchNorm',
                        choices=['BatchNorm', 'LayerNorm'],
                        help='Normalization type used in SST short-range transformer')
    parser.add_argument('--attn_dropout', type=float, default=0.0,
                        help='Attention dropout used in SST short-range transformer')
    parser.add_argument('--act', type=str, default='gelu',
                        choices=['relu', 'gelu'],
                        help='Activation function used inside SST modules')
    parser.add_argument('--key_padding_mask', type=str, default='auto',
                        help='Key padding mask option used in SST attention')
    parser.add_argument('--padding_var', type=int, default=None,
                        help='Padding variable index used in SST; default None')
    parser.add_argument('--res_attention', type=int, default=1,
                        help='Use residual attention in SST short-range transformer (1=True, 0=False)')
    parser.add_argument('--pre_norm', type=int, default=0,
                        help='Use pre-norm in SST transformer blocks (1=True, 0=False)')
    parser.add_argument('--pe', type=str, default='zeros',
                        help='Positional encoding type in SST')
    parser.add_argument('--learn_pe', type=int, default=1,
                        help='Whether SST positional encoding is learnable (1=True, 0=False)')
    parser.add_argument('--pretrain_head', type=int, default=0,
                        help='Use pretrain head in SST (1=True, 0=False)')
    parser.add_argument('--verbose', type=int, default=0,
                        help='Verbose mode for SST (1=True, 0=False)')

    # -----------------------------
    # MTLinear
    # -----------------------------
    parser.add_argument('--mtlinear_cluster_dist', type=float, default=1,
                        help='Cluster number or distance threshold for MTLinear. Default 1 means one cluster.')

    parser.add_argument('--mtlinear_layer_type', type=str, default='Linear',
                        choices=['Linear', 'DLinear', 'NLinear', 'RLinear'],
                        help='Base linear layer type used in MTLinear.')

    # ============================================================
    # MTST 
    # ============================================================

    parser.add_argument('--mtst_n_branches', type=int, default=3,
                        help='Number of multi-resolution branches in MTST')

    parser.add_argument('--mtst_patch_len_ls', type=int, nargs='+', default=[4, 8, 16],
                        help='Patch lengths for MTST multi-resolution branches, e.g. --mtst_patch_len_ls 4 8 16')

    parser.add_argument('--mtst_stride_ls', type=int, nargs='+', default=[2, 4, 8],
                        help='Patch strides for MTST multi-resolution branches, e.g. --mtst_stride_ls 2 4 8')

    parser.add_argument('--mtst_rel_pe', type=str, default=None,
                        help='Relative positional encoding type in MTST. Options: None, rel_sin')

    parser.add_argument('--mtst_res_attn', type=int, default=0,
                        help='Use residual attention inside MTST (1=True, 0=False)')

    parser.add_argument('--mtst_return_all', type=int, default=0,
                        help='Return prediction, branch outputs, and attention maps for MTST debugging (1=True, 0=False)')


    # -----------------------------
    # SAMBA
    # -----------------------------
    parser.add_argument('--d_state1', type=int, default=16,
                        help='State dimension for SAMBA temporal encoder')
    parser.add_argument('--d_state2', type=int, default=16,
                        help='State dimension for SAMBA variable encoder')
    parser.add_argument('--use_act', type=int, default=0,
                        help='Use activation in SAMBA convolution branch: 1=True, 0=False')

    # -----------------------------
    # TimeMachine
    # -----------------------------
    parser.add_argument('--n1', type=int, default=128,
                        help='First hidden length in TimeMachine')
    parser.add_argument('--n2', type=int, default=256,
                        help='Second hidden length in TimeMachine')
    parser.add_argument('--dconv', type=int, default=4,
                        help='Convolution kernel size in Mamba blocks of TimeMachine')
    parser.add_argument('--e_fact', type=int, default=2,
                        help='Expansion factor in Mamba blocks of TimeMachine')
    parser.add_argument('--ch_ind', type=int, default=1,
                        help='Use channel-independent mode in TimeMachine (1=True, 0=False)')
    parser.add_argument('--residual', type=int, default=1,
                        help='Use residual connections in TimeMachine (1=True, 0=False)')


    # -----------------------------
    # ChronoOrg
    # -----------------------------
    parser.add_argument('--chronoorg_level', type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        help='ChronoOrg list preset level: 0=default, 1=xxtiny, 2=xtiny, 3=tiny, 4=tiny2, 5=small, 6=small2, 7=base, 8=large, 9=large_384, 10=large_512')

    parser.add_argument('--num_heads', nargs='+', type=int, default=[2, 4, 8],
                        help='Number of attention heads in each ChronoOrg stage')

    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='Expansion ratio for FFN hidden dimension in ChronoOrg')

    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'last'],
                        help='Pooling type for ChronoOrg prediction head')

    parser.add_argument('--tokenizer_bias', type=int, default=1,
                        help='Use bias term in temporal tokenizer (1=True, 0=False)')

    parser.add_argument('--stage_dims', nargs='+', type=int, default=None,
                        help='Hidden dimensions for each ChronoOrg stage; if None, progressive doubling from d_model is used')

    parser.add_argument('--ffn_hidden_dim', nargs='+', type=int, default=None,
                        help='FFN hidden dimensions for each ChronoOrg stage; if None, computed by mlp_ratio')

    parser.add_argument('--depths', type=int, nargs='+', default=[2, 2, 2],
                    help='Number of blocks in each stage')

    parser.add_argument('--window_size', type=int, nargs='+', default=[8, 8, 4], help='Window size for each stage, e.g. --window_size 8 8 4')

    parser.add_argument('--global_tokens', nargs='+', type=int, default=None,
                        help='Number of global tokens for each ChronoOrg stage; if None, use window_size of each stage')

    parser.add_argument('--stage_use_global', nargs='+', type=str, default=None,
                        help='Whether to use global attention in each block of each stage, e.g. 0,1 0,1 0,1,0,1')

    parser.add_argument('--qkv_bias', type=int, default=1,
                        help='Use bias term in QKV projection (1=True, 0=False)')

    parser.add_argument('--layer_scale', type=float, default=None,
                        help='Layer scale initialization value; default None')

    parser.add_argument('--downsample', type=int, default=1,
                        help='Use downsampling between adjacent ChronoOrg stages (1=True, 0=False)')

    parser.add_argument('--period', type=int, default=24,
                        help='Temporal period prior used in ChronoOrg')

    parser.add_argument('--revin_eps', type=float, default=1e-5,
                        help='Epsilon value in RevIN normalization')


    # -----------------------------
    # Paimon
    # -----------------------------
    parser.add_argument('--use_patch_spectral_embedding', type=int, default=1,
                        help='Use Patch-Aware Spectral Embedding in Paimon (1=True, 0=False)')

    parser.add_argument('--pase_low_ratio', type=float, default=0.35,
                        help='Low-frequency band ratio in Patch-Aware Spectral Embedding')

    parser.add_argument('--pase_mid_ratio', type=float, default=0.70,
                        help='Middle-frequency band ratio in Patch-Aware Spectral Embedding')

    parser.add_argument('--pase_gate_hidden_dim', type=int, default=None,
                        help='Hidden dimension of the adaptive spectral gate in Patch-Aware Spectral Embedding; None means auto')

    parser.add_argument('--pase_use_adaptive_gate', type=int, default=1,
                        help='Use sample-adaptive spectral gate in Patch-Aware Spectral Embedding (1=True, 0=False)')

    parser.add_argument('--pase_use_learnable_freq_bias', type=int, default=1,
                        help='Use learnable frequency bias in Patch-Aware Spectral Embedding (1=True, 0=False)')

    parser.add_argument('--pase_residual_scale_init', type=float, default=-1.0,
                        help='Initial residual scale for Patch-Aware Spectral Embedding')

    parser.add_argument('--pase_gate_temperature', type=float, default=1.0,
                        help='Temperature of the spectral gate in Patch-Aware Spectral Embedding')

    parser.add_argument('--pase_use_patch_norm', type=int, default=1,
                        help='Use normalization before patch projection in Patch-Aware Spectral Embedding (1=True, 0=False)')


    # Period-aware Patch Filter
    parser.add_argument('--period_rf_scale', type=float, default=1.0,
                        help='Receptive-field scaling factor for period-aware patch filter')

    parser.add_argument('--filter_kernels', type=str, default=None,
                        help='Manual patch filter kernels, e.g., "3,5,9"; None/auto means infer from period')

    parser.add_argument('--filter_kernel_target_num', type=int, default=3,
                        help='Target number of inferred patch filter kernels')

    parser.add_argument('--filter_gate_temperature', type=float, default=1.0,
                        help='Temperature of the adaptive patch filter gate')


    # Attention replacement
    parser.add_argument('--attention_replacement', type=str, default='filter1d',
                        choices=['filter1d', 'attention'],
                        help='Token interaction module in Paimon: filter1d replaces self-attention; attention uses standard self-attention for comparison')

    parser.add_argument('--filter_init_scale', type=float, default=0.02,
                        help='Initialization scale of learnable Filter1D frequency weights')

    parser.add_argument('--use_filter_gate', type=int, default=1,
                        help='Use learnable frequency gate in Filter1D (1=True, 0=False)')

    # -----------------------------
    # Logging / Saving
    # -----------------------------
    parser.add_argument('--save_dir', type=str, default='result_forecast',
                        help='Directory for saving results and checkpoints')

    parser.add_argument(
    "--decomp_method",
    type=str,
    default="ours",
    choices=["ema", "dema", "ours", "ca_dema", "ca-dema", "cg_cdema", "cg-cdema", "beda"],
    help="Decomposition method: ema, dema, ours/ca_dema/cg_cdema, or beda.")
    # TCT decomposition ablation
    parser.add_argument(
        "--decomp_type",
        type=str,
        default="ema",
        choices=["ma", "ema", "dema", "learnable"],
        help="trend decomposition type: ma, ema, dema, learnable"
    )

    parser.add_argument(
        "--use_increment",
        type=int,
        default=0,
        help="whether to use second-order curvature-aware trend control"
    )

    parser.add_argument(
        "--tct_scale",
        type=float,
        default=2.0,
        help="curvature gate sensitivity for TCT"
    )

    parser.add_argument(
        "--moving_avg_kernel",
        type=int,
        default=25,
        help="kernel size for moving-average decomposition"
    )

    parser.add_argument(
        "--learnable_kernel",
        type=int,
        default=25,
        help="kernel size for learnable convolutional decomposition"
    )

    return parser.parse_args()


# ============================================================
# Utils
# ============================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_transform_3d(arr_3d: np.ndarray, scaler):
    """
    arr_3d: (B, H, C)
    scaler: sklearn StandardScaler
    """
    if scaler is None:
        return arr_3d

    B, H, C = arr_3d.shape
    arr_2d = arr_3d.reshape(-1, C)
    arr_2d = scaler.inverse_transform(arr_2d)
    return arr_2d.reshape(B, H, C)


def compute_metrics(preds: np.ndarray, gts: np.ndarray):
    """
    preds, gts: (N, H, C), already in original scale
    """
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
    """
    Unified alignment:
    keep only the last pred_len on time dimension.
    tensor: [B, T, ...] or None
    """
    if tensor is None:
        return None
    return tensor[:, -pred_len:, ...]


def build_decoder_input(y, label_len, pred_len, device):
    """
    y: [B, label_len + pred_len, C]
    dec_inp: [B, label_len + pred_len, C]
        = [known history tail(label_len), zeros(pred_len)]
    """
    if y is None:
        return None

    if y.size(1) < label_len + pred_len:
        raise ValueError(
            f"y length {y.size(1)} is smaller than label_len + pred_len = {label_len + pred_len}"
        )

    dec_zeros = torch.zeros_like(y[:, -pred_len:, :]).float()
    dec_inp = torch.cat([y[:, :label_len, :], dec_zeros], dim=1).float().to(device)
    return dec_inp


def is_decoder_model(model_name: str):
    """
    Models in this list will use:
        model(x, x_mark, dec_inp, y_mark)
    Modify this list according to your actual decoder-based models.
    """
    decoder_models = {
        "transformer",
        "autoformer",
        "informer",
        "fedformer",
        "itransformer",
        # "gcvit",
    }
    return model_name.lower() in decoder_models


# ============================================================
# Train
# ============================================================

def train(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    # args = process_chronoorg_args(args)
    # --------------------------------------------------------
    # 1) Load data
    # --------------------------------------------------------
    train_loader, val_loader, test_loader, input_dim, scaler = load_data(args)

    print(f"[INFO] Dataset={args.data_type}, input_dim={input_dim}")
    print(f"[INFO] Device={device}")

    # --------------------------------------------------------
    # 2) Build model
    # --------------------------------------------------------
    args.time_dim = infer_time_dim(args.freq, args.timeenc)
    model = get_model(args, input_dim=input_dim).to(device)
    print("Model device:", next(model.parameters()).device)

    # --------------------------------------------------------
    # 3) Loss & Optimizer
    # --------------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # --------------------------------------------------------
    # 4) Paths
    # --------------------------------------------------------
    best_model_path = os.path.join(args.save_dir, build_best_model_name(args))

    # --------------------------------------------------------
    # 5) Training loop
    # --------------------------------------------------------
    best_val_loss = float("inf")
    early_stop_cnt = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for idx, batch in enumerate(train_loader):
            x, y, x_mark, y_mark, cycle = batch
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if is_decoder_model(args.model_name):
                dec_inp = build_decoder_input(y, args.label_len, args.pred_len, device)
                output = model(x, x_mark, dec_inp, y_mark)
            elif args.model_name.lower() == 'tqnet':
                cycle = cycle.to(device)
                output = model(x, cycle)
            else:
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

        # ----------------------------------------------------
        # 6) Validation
        # ----------------------------------------------------
        model.eval()

        val_loss = 0.0
        val_start_time = time.time()

        with torch.no_grad():
            for batch in val_loader:
                x, y, x_mark, y_mark, cycle = batch
                x_mark = x_mark.to(device)
                y_mark = y_mark.to(device)
                x = x.to(device)
                y = y.to(device)

                if is_decoder_model(args.model_name):
                    dec_inp = build_decoder_input(y, args.label_len, args.pred_len, device)
                    output = model(x, x_mark, dec_inp, y_mark)
                elif args.model_name.lower() == 'tqnet':
                    cycle = cycle.to(device)
                    output = model(x, cycle)
                else:
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
        # ----------------------------------------------------
        # 7) Early stopping
        # ----------------------------------------------------
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

    # --------------------------------------------------------
    # 8) Test
    # --------------------------------------------------------
    print("[INFO] Loading best model for test...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds = []
    gts = []

    with torch.no_grad():
        for batch in test_loader:
            x, y, x_mark, y_mark, cycle = batch
            x_mark = x_mark.to(device)
            y_mark = y_mark.to(device)
            x = x.to(device)
            y = y.to(device)

            if is_decoder_model(args.model_name):
                dec_inp = build_decoder_input(y, args.label_len, args.pred_len, device)
                output = model(x, x_mark, dec_inp, y_mark)
            elif args.model_name.lower() == 'tqnet':
                cycle = cycle.to(device)
                output = model(x, cycle)
            else:
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

    # preds_inv = inverse_transform_3d(preds, scaler)
    # gts_inv = inverse_transform_3d(gts, scaler)
    # metrics = compute_metrics(preds_inv, gts_inv)

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


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = parse_args()
    train(args)