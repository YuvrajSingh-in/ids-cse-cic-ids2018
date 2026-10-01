"""
data_loader.py
──────────────
Responsible solely for reading raw CSV files from disk and
returning a single, concatenated DataFrame.

Design decisions:
- low_memory=False prevents dtype inference warnings on large CSVs.
- Column names are stripped of leading/trailing whitespace because
  CICFlowMeter outputs are inconsistent across files.
- No transformation is performed here; the loader is intentionally
  "dumb" so preprocessing stays isolated and testable.
"""

import os
import pandas as pd


# Default file list relative to the project root.
# Adjust paths if files are stored elsewhere.
DEFAULT_FILES = [
    os.path.join("data", "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"),
    os.path.join("data", "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv"),
    os.path.join("data", "Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv"),
]


def load_data(file_paths: list[str] = DEFAULT_FILES) -> pd.DataFrame:
    """
    Load and concatenate one or more CICFlowMeter CSV files.

    Parameters
    ----------
    file_paths : list of str
        Paths to CSV files. Defaults to the three IDS2018 files.

    Returns
    -------
    pd.DataFrame
        Concatenated raw DataFrame with whitespace-cleaned column names.

    Raises
    ------
    FileNotFoundError
        If any path in file_paths does not exist on disk.
    """
    frames: list[pd.DataFrame] = []

    for path in file_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Dataset file not found: '{path}'\n"
                f"Ensure CSVs are placed inside the 'data/' directory."
            )
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        frames.append(df)
        print(f"  [loader] {os.path.basename(path)}: {len(df):,} rows, {df.shape[1]} cols")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  [loader] Combined: {len(combined):,} rows total\n")
    return combined
