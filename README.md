# KDN-DRL: Knowledge Defined Networking with Deep Reinforcement Learning

This project implements a Deep Reinforcement Learning (DRL) approach for network routing optimization, leveraging Knowledge Defined Networking (KDN) concepts. It supports training and benchmarking of agents (e.g., MaskPPO) in simulated network environments (Deflection, KDN).

## Datasets

This project uses network modeling datasets (NSFNet, GEANT2, etc.) for training and benchmarking.

1. **Download**: Visit [Knowledge-Defined Networking Training Datasets](https://knowledgedefinednetworking.org/) to download the required datasets (TFRecords format).
2. **Extraction**: Unzip the downloaded datasets into the `data/` directory.
3. **Structure**: Ensure the data is organized as follows:

```bash
data/
├── nsfnetbw/  # Example dataset
└── geant2bw/  # Example dataset
```

## Usage

The primary entry point is `main.py`, which handles both training and benchmarking phases sequentially.

### Quick Start

To run the default training and benchmarking pipeline:

```bash
uv run python main.py
```

### Configuration Arguments

You can customize the execution using command-line arguments.

#### Common Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--tfrecords_dir` | str | `data/nsfnetbw` | Path to dataset root directory |
| `--traffic_intensity` | int | `15` | Traffic intensity filter level |
| `--dataset_name` | str | `None` | (Optional) Dataset name |
| `--model_type` | str | `MaskPPO` | Model type identifier |
| `--gnn_type` | str | `none` | GNN feature extractor: `gcn`, `gat`, or `none` |
| `--device` | str | `Auto` | Device to run on (`cpu`, `cuda`, `mps`) |
| `--seed` | int | `42` | Random seed for reproducibility |

#### Training Configuration

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--total_timesteps` | int | `100_000` | Total training timesteps |
| `--n_envs` | int | `1` | Number of parallel environments |
| `--log_interval` | int | `1` | Interval for logging metrics |
| `--model_path` | str | `final_model` | Filename for saving the trained model |
| `--data_filter` | str | `all` | Data filtering strategy: `all`, `sp`, `optimal` |
| `--min_hops` | int | `1` | Minimum shortest path hops for filtering |

#### Environment & Benchmark Configuration

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--run_benchmark` | bool | `True` | Whether to run benchmark after training |
| `--num_samples` | int | `100` | Number of samples for benchmarking |
| `--env_type` | str | `base` | Environment type: `base` (DeflectionEnv), `masked` (MaskedDeflectionEnv) |

### Examples

#### Naive Deflection

```bash
uv run python main.py --total_timesteps 1024 --num_samples 100 --env_type base --data_filter optimal
```

#### Masked Env

```bash
uv run python main.py --total_timesteps 1024 --num_samples 100 --env_type masked --data_filter optimal
```

#### GCN Masked Env

```bash
uv run python main.py --total_timesteps 1024 --num_samples 100 --env_type masked --gnn_type gcn --data_filter optimal
```
