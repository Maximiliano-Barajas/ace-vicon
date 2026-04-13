import os

import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging

from prepare_data import load_prepared_serves

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "plotting", "markers", "unmarked_edited"
)
OUT_PATH = os.path.join(os.path.dirname(__file__), "barycenter.npy")


def compute_barycenter(dirpath=DATA_DIR, out_path=OUT_PATH):
    """Load all serves, compute the DTW barycenter, and save it.

    Args:
        dirpath: folder of multi-marker Vicon CSVs (unmarked_edited)
        out_path: .npy file to write the barycenter to

    Returns:
        barycenter as np.ndarray of shape (n_frames, n_features)
    """
    arrays = load_prepared_serves(dirpath)
    print(f"Loaded {len(arrays)} valid serves")

    # Pass the list of arrays directly — to_time_series_dataset pads with NaN
    # which causes dtw_barycenter_averaging to return None on the full dataset
    barycenter = dtw_barycenter_averaging(arrays)
    print(f"Barycenter shape: {barycenter.shape}")

    np.save(out_path, barycenter)
    print(f"Saved to {out_path}")

    return barycenter


if __name__ == "__main__":
    compute_barycenter()
