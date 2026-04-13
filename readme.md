# V-TrustFL: Federated Learning with Neuro-Fuzzy Dynamic Trust for Continuous Authentication

A robust Continuous Authentication (CA) framework built on **Federated Learning (FL)** combined with a **Neuro-Fuzzy** inference system to evaluate dynamic trust scores in Zero-Trust environments.

## Overview

This project implements an advanced AI architecture designed for mobile security. It authenticates users based on behavioral biometrics (Accelerometer and Gyroscope data) without compromising privacy, as raw data never leaves the local device.

### Key Features:
- **Hybrid CNN-LSTM Backbone:** Captures both spatial features and long-term temporal dependencies from sensor streams.
- **Proposed Neuro-Fuzzy Aggregation:** A novel FL weight aggregation mechanism that considers both AI model confidence and sensor noise levels.
- **Zero-Trust Simulation:** Real-time simulation of authentication sessions with dynamic trust decay and recovery curves.
- **Comparative Analysis:** Built-in benchmarking against state-of-the-art methods: FedAvg, FedProx, and Centralized training.

## Project Structure

| File | Description |
|:--- |:--- |
| `config.py` | System configurations (Hyperparameters, User count, Dataset paths). |
| `models.py` | Implementation of CNN-LSTM architectures and the Neuro-Fuzzy layer. |
| `engine.py` | Core execution logic for Baseline and Proposed FL training rounds. |
| `evaluator.py` | Metric calculations (EER, Accuracy, FAR, FRR) and visualization tools. |
| `plot_fuzzy_system.py` | Visualizes Membership Functions (MFs) and the 3D Fuzzy control surface. |
| `run_all.py` | Master script to execute the entire pipeline (Preprocessing -> Training -> Evaluation). |
| `graph.py` | Generates comparison charts between V-TrustFL and existing SOTA research. |

## Getting Started

### 1. Prerequisites
- Python 3.8 or higher
- PyTorch (with CUDA support recommended)
- Core libraries: `pip install torch numpy pandas scikit-learn matplotlib seaborn openpyxl`

### 2. Dataset Configuration
Update the **HMOG Dataset** path in `config.py`:
```python
DATASET_DIR_PATH = r"C:\Your\Path\To\HMOG_Dataset"