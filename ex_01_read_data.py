import numpy as np
import pandas as pd
from pathlib import Path


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load and preprocess data from a CSV file. Remove rows with unlabeled data.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    df = remove_unlabeled_data(df)
    df = df.dropna().reset_index(drop=True)
    if df.empty:
        raise ValueError("No data available after preprocessing.")
    return df


def remove_unlabeled_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with unlabeled data (where labels == -1).
    """
    if 'labels' not in data.columns:
        raise KeyError("Input DataFrame must contain a 'labels' column.")
    return data[data['labels'] != -1].reset_index(drop=True)


def convert_to_np(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate labels, exp_ids and stack current/voltage into shape (n_samples, timesteps, 2).
    """
    # Check required columns
    for col in ('labels', 'exp_ids'):
        if col not in data.columns:
            raise KeyError(f"Missing required column: {col}")
    labels = data['labels'].to_numpy()
    exp_ids = data['exp_ids'].to_numpy()

    # Identify and sort feature columns
    current_cols = sorted([c for c in data.columns if c.startswith('I')])
    voltage_cols = sorted([c for c in data.columns if c.startswith('V')])
    if not current_cols or not voltage_cols:
        raise KeyError("Current or voltage feature columns not found.")

    currents = data[current_cols].to_numpy(dtype=float)
    voltages = data[voltage_cols].to_numpy(dtype=float)
    if currents.shape[1] != voltages.shape[1]:
        raise ValueError("Mismatch in number of timesteps between currents and voltages.")

    # Stack last axis: currents and voltages are two feature-dimensions
    combined = np.stack((currents, voltages), axis=-1)
    return labels, exp_ids, combined


def create_sliding_windows_first_dim(data: np.ndarray, sequence_length: int) -> np.ndarray:
    """
    Create sliding windows over the sample dimension: output shape (n_windows, seq_len, timesteps, features).
    """
    from numpy.lib.stride_tricks import sliding_window_view
    if data.ndim != 3:
        raise ValueError("Input data must be 3D.")
    n_samples, timesteps, features = data.shape
    if sequence_length < 1 or sequence_length > n_samples:
        raise ValueError(f"sequence_length must be between 1 and {n_samples}.")
    return sliding_window_view(data, window_shape=sequence_length, axis=0)


def get_welding_data(
    path: Path,
    n_samples: int | None = None,
    return_sequences: bool = False,
    sequence_length: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data (CSV or cached .npy), optionally generate flattened sequences.
    """
    base = path.stem
    data_file = path.parent / f"{base}_data.npy"
    labels_file = path.parent / f"{base}_labels.npy"
    expids_file = path.parent / f"{base}_exp_ids.npy"

    if data_file.exists() and labels_file.exists() and expids_file.exists():
        data = np.load(data_file)
        labels = np.load(labels_file)
        exp_ids = np.load(expids_file)
    else:
        df = load_data(path)
        labels, exp_ids, data = convert_to_np(df)
        np.save(data_file, data)
        np.save(labels_file, labels)
        np.save(expids_file, exp_ids)

    if return_sequences:
        # Build sequences manually to ensure correct shape and order
        n_samples_total, timesteps, features = data.shape
        n_windows = n_samples_total - sequence_length + 1
        if n_windows < 1:
            raise ValueError(f"sequence_length {sequence_length} too large for number of samples {n_samples_total}.")
        seq_data = []
        seq_labels = []
        seq_exp = []
        for start in range(n_windows):
            window = data[start:start+sequence_length]  # shape (sequence_length, timesteps, features)
            # flatten sequence and time dims
            flattened = window.reshape(sequence_length * timesteps, features)
            seq_data.append(flattened)
            seq_labels.append(labels[start:start+sequence_length])
            seq_exp.append(exp_ids[start:start+sequence_length])
        data = np.stack(seq_data, axis=0)    # (n_windows, sequence_length*timesteps, features)
        labels = np.stack(seq_labels, axis=0) # (n_windows, sequence_length)
        exp_ids = np.stack(seq_exp, axis=0)  # (n_windows, sequence_length)
    # random sampling
    total = data.shape[0]
    if n_samples is not None:
        if n_samples < 0:
            raise ValueError("n_samples must be non-negative.")
        k = min(n_samples, total)
        idx = np.random.choice(total, size=k, replace=False)
        data = data[idx]
        labels = labels[idx]
        exp_ids = exp_ids[idx]

    return data, labels, exp_ids
