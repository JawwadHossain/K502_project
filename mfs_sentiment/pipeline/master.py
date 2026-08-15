import os
import pandas as pd
from pathlib import Path


from mfs_sentiment.config import (
    DATASET_PATH_DICT,
    REQUIRED_MASTER_CSV_COLUMNS,
    REQUIRED_SCORED_MASTER_CSV_COLUMNS,
    get_run_paths,
)
from mfs_sentiment.preprocessing.pipeline import build_standarized_csv
from mfs_sentiment.scoring.gap import generate_gap_scores
from mfs_sentiment.scoring.sentiment import generate_sentiment_scores


def build_master_csv(only_english: bool = False) -> pd.DataFrame:
    """Build the combined, unscored master CSV from all apps' cleaned reviews.

    Cleans each app's raw csv, concatenates the results, and saves the combined
    dataset to the current mode's result folder. Sentiment and gap scores are NOT
    computed here — run score_master_csv() afterward.
    """
    run_paths = get_run_paths(only_english)
    cleaned_frames = []
    for _, raw_path in DATASET_PATH_DICT.items():
        cleaned_path = build_standarized_csv(raw_path, only_english=only_english)
        cleaned_frames.append(pd.read_csv(cleaned_path))

    master_df = pd.concat(cleaned_frames, ignore_index=True)
    master_df.to_csv(run_paths["master_csv_path"], index=False)
    return master_df


def load_master_csv(path: Path|str, required_columns: list = REQUIRED_SCORED_MASTER_CSV_COLUMNS) -> pd.DataFrame | None:
    """
    Load and validate a master CSV file against a set of required columns.

    Parameters
    ----------
    path : str
        Path to the master CSV file to load.
    required_columns : list of str, optional
        Columns that must be present. Default is REQUIRED_SCORED_MASTER_CSV_COLUMNS
        (use REQUIRED_MASTER_CSV_COLUMNS for the pre-scoring master csv).

    Returns
    -------
    pd.DataFrame or None
        The loaded dataframe, or None if any required column is missing.
    """
    df = pd.read_csv(path)
    has_required_columns = set(required_columns).issubset(df.columns)
    if not has_required_columns:
        print("Error: The csv file must have the following columns: ", required_columns)
        return None
    return df


def score_master_csv(master_csv_path: Path| str | None = None, only_english: bool = False) -> pd.DataFrame:
    """
    Compute sentiment and gap scores for the master CSV and save the result
    """
    if master_csv_path is None:
        master_csv_path = get_run_paths(only_english)["master_csv_path"]

    scored_path = generate_sentiment_scores(
        master_csv_path, # type: ignore
        output_path=get_run_paths(only_english)["scored_master_csv_path"],
    )
    return generate_gap_scores(scored_path)