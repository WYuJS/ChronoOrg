# ChronoOrg

PyTorch implementation of ChronoOrg for multivariate time-series forecasting.

This release contains the main forecasting model and example configurations for
PEMS03, PEMS04, PEMS07, PEMS08, and Solar. The complete set of paper configurations
and the separately implemented ablation models are not included. These examples
have not been rerun end to end to verify the paper's reported numerical results.

## Setup

Install Python dependencies in your environment:

```bash
python3 -m pip install -r requirements.txt
```

The experiment scripts require Bash and a `python3` command. For GPU training,
install a PyTorch build appropriate for your CUDA environment.

## Data

Place the datasets in `data/` at the repository root. Data files are not bundled.

| Dataset | Expected filename | Example script |
| --- | --- | --- |
| PEMS03 | `data/PEMS03.npz` | `experiments/Pems03.sh` |
| PEMS04 | `data/pems04.npz` | `experiments/Pems04.sh` |
| PEMS07 | `data/pems07.npz` | `experiments/Pems07.sh` |
| PEMS08 | `data/pems08.npz` | `experiments/Pems08.sh` |
| Solar | `data/solar_AL.txt` | `experiments/Solar.sh` |

## Run

From the repository root, run one of the examples:

```bash
bash experiments/Pems03.sh
bash experiments/Pems04.sh
bash experiments/Pems07.sh
bash experiments/Pems08.sh
bash experiments/Solar.sh
```

Each script evaluates four forecast horizons. Logs and summaries are written to
`experiments/logs/` and `experiments/TXT/`; checkpoints are written to
`result_forecast/`. For available model and training options:

```bash
python3 train.py --help
```

## Implementation details

- The model is defined in `model/ChronoOrg.py`, with modules named after the paper:
  Offset-Aware Temporal Injection, Bounded Temporal Interaction, Sequence Context
  Encoder, Context-Conditioned Temporal Attention, and Temporal Merge.
- Dataset normalization is fitted on the training partition. RevIN is additionally
  enabled by default; its affine and centering settings are specified by the
  configuration. Metrics are computed on the dataset-standardized scale.
- Training uses Adam and MSE loss, with validation-based early stopping. The
  checkpoint with the lowest validation MSE is restored for test evaluation.
- The current loaders use `drop_last=True`, including validation and test. Test
  sample selection therefore depends on the evaluation batch size. Comparisons
  should use matching evaluated samples.

Datasets, checkpoints, generated logs, and local backups are excluded from Git.
