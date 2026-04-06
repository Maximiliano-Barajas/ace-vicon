import pandas as pd

def load_unmarked_csv(filepath):
    """
    Load an unmarked marker CSV file (Motive export format).

    Structure:
      Row 0: marker names (every 3 cols starting at col 2)
      Row 1: TX/TY/TZ labels
      Row 2: units (mm) — skipped
      Row 3+: data — col 0 = frame, col 1 = sub frame (ignored),
               then groups of 3 columns = TX, TY, TZ per marker

    Returns a dict:
      {
        'frames': np.ndarray of frame numbers,
        '<marker_name>': {'TX': np.ndarray, 'TY': np.ndarray, 'TZ': np.ndarray},
        ...
      }
    """
    raw = pd.read_csv(filepath, header=None, dtype=str)

    # Extract marker names from row 0 (col 2, 5, 8, ... — every 3rd col)
    n_cols = raw.shape[1]
    n_markers = (n_cols - 2) // 3
    marker_names = []
    for i in range(n_markers):
        col_idx = 2 + i * 3
        name = raw.iat[0, col_idx]
        if pd.isna(name) or str(name).strip() == '':
            name = f'Marker_{i + 1}'
        marker_names.append(str(name).strip())

    # Skip the 3 header rows; remaining rows are data
    data = raw.iloc[3:].reset_index(drop=True)

    frames = pd.to_numeric(data.iloc[:, 0], errors='coerce').values

    result = {'frames': frames}
    for i, name in enumerate(marker_names):
        c = 2 + i * 3
        result[name] = {
            'TX': pd.to_numeric(data.iloc[:, c],     errors='coerce').values,
            'TY': pd.to_numeric(data.iloc[:, c + 1], errors='coerce').values,
            'TZ': pd.to_numeric(data.iloc[:, c + 2], errors='coerce').values,
        }

    return result


if __name__ == '__main__':
    import os

    csv_path = os.path.join(os.path.dirname(__file__), 'serve3.csv')
    data = load_unmarked_csv(csv_path)

    print(f"Frames: {len(data['frames'])} ({data['frames'][0]} - {data['frames'][-1]})")
    print(f"Markers ({len(data) - 1}):")
    for name in list(data.keys())[1:]:
        tx = data[name]['TX']
        print(f"  {name}  TX[0]={tx[0]:.3f}  shape={tx.shape}")
