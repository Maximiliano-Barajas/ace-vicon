import pandas as pd
import os

NAME_MAP = {
    "chest": "chest",
    "forehead": "head",
    "leftelbow": "left_elbow",
    "leftfoot": "left_foot",
    "lefthand": "left_hand",
    "lefthip": "left_hip",
    "leftknee": "left_knee",
    "leftshoulder": "left_shoulder",
    "rightelbow": "right_elbow",
    "rightfoot": "right_foot",
    "righthand": "right_hand",
    "righthip": "right_hip",
    "rightknee": "right_knee",
    "rightshoulder": "right_shoulder",
}


def load_joints(folder_path):
    joints = pd.read_csv(
        folder_path + "/joints.csv",
        skiprows=3,
        names=["frame", "subframe", "tx", "ty", "tz"],
    )
    print(joints.head())
    return joints


def load_all_markers(folder_path):
    """Load all marker CSVs and return a dict of {joint_name: DataFrame}."""
    markers = {}
    for csv_file in os.listdir(folder_path):
        if not csv_file.endswith(".csv"):
            continue
        stem = csv_file.replace(".csv", "").lower()
        joint_name = NAME_MAP.get(stem)
        if joint_name is None:
            print(f"Warning: no mapping for {csv_file}, skipping")
            continue
        path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(
            path, skiprows=3, names=["frame", "subframe", "tx", "ty", "tz"]
        )
        markers[joint_name] = df
    return markers


if __name__ == "__main__":
    markers_folder = "markers"
    for csv_file in os.listdir(markers_folder):
        if csv_file.endswith(".csv"):
            path = os.path.join(markers_folder, csv_file)
            print(f"\n--- {csv_file} ---")
            df = pd.read_csv(
                path, skiprows=3, names=["frame", "subframe", "tx", "ty", "tz"]
            )
            print(df.head())
