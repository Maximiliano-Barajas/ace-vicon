import pandas as pd


def load_vicon_csv(file_path):
    """
    Loads Vicon CSV motion capture data into a pandas DataFrame.
    Skips multi-row header formatting.
    """
    try:
        # Skip the first 3 rows
        data = pd.read_csv(file_path, skiprows=3)

        # Rename columns manually for clarity
        data.columns = ["Frame", "SubFrame", "TX", "TY", "TZ", "RX", "RY", "RZ"]

        print("File loaded successfully.")
        print(data.head())
        return data

    except Exception as e:
        print(f"Error loading file: {e}")
        return None
