from pathlib import Path
from data.vicon_loader import load_vicon_csv


def main():
    # Get project root automatically
    project_root = Path(__file__).resolve().parent.parent

    file_path = project_root / "data" / "raw" / "max_swing1.csv"

    data = load_vicon_csv(file_path)

    if data is not None:
        print(f"Loaded {len(data)} frames.")


if __name__ == "__main__":
    main()
