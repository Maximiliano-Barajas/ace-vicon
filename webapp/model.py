import numpy as np
import pandas as pd


def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df.select_dtypes(include=[np.number]).values


def build_reference_model(reference_files):
    sequences = [load_csv(f) for f in reference_files]

    min_len = min(seq.shape[0] for seq in sequences)
    sequences = [seq[:min_len] for seq in sequences]

    stacked = np.stack(sequences)

    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0) + 1e-6  # avoid divide by zero

    return mean, std


def compute_similarity(user_file, reference_files):
    user = load_csv(user_file)

    mean, std = build_reference_model(reference_files)

    min_len = min(user.shape[0], mean.shape[0])
    user = user[:min_len]
    mean = mean[:min_len]
    std = std[:min_len]

    # Gaussian likelihood-style scoring
    z = (user - mean) / std
    score = np.exp(-0.5 * np.mean(z**2))

    return float(score * 100)
