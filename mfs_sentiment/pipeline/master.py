import os
import pandas as pd
from pathlib import Path


from mfs_sentiment.config import (
    DATASET_PATH_DICT,
    MASTER_CSV_PATH,
    REQUIRED_MASTER_CSV_COLUMNS,
    REQUIRED_SCORED_MASTER_CSV_COLUMNS,
    SCORED_MASTER_CSV_PATH,
)
from mfs_sentiment.preprocessing.pipeline import build_standarized_csv
from mfs_sentiment.scoring.gap import generate_gap_scores
from mfs_sentiment.scoring.sentiment import generate_sentiment_scores


def build_master_csv(only_english: bool = False) -> pd.DataFrame:
    """Build the combined, unscored master CSV from all apps' cleaned reviews.

    Cleans each app's raw csv (see build_standarized_csv), concatenates the
    results, and saves the combined dataset to MASTER_CSV_PATH. Sentiment and
    gap scores are NOT computed here — run score_master_csv() afterward.

    Parameters
    ----------
    only_english : bool, optional
        Passed through to build_standarized_csv for each app. Default is False.

    Returns
    -------
    pd.DataFrame
        The combined, cleaned (unscored) master dataframe.
    """
    cleaned_frames = []
    for _, raw_path in DATASET_PATH_DICT.items():
        cleaned_path = build_standarized_csv(raw_path, only_english=only_english)
        cleaned_frames.append(pd.read_csv(cleaned_path))

    master_df = pd.concat(cleaned_frames, ignore_index=True)
    master_df.to_csv(MASTER_CSV_PATH, index=False)
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


def score_master_csv(master_csv_path: Path|str = MASTER_CSV_PATH) -> pd.DataFrame:
    """Compute sentiment and gap scores for the master CSV and save the result.

    Loads the unscored master csv, runs VADER sentiment scoring on
    'review_text_clean', computes gap scores and mismatch labels, and writes
    the fully scored dataset to SCORED_MASTER_CSV_PATH.

    Parameters
    ----------
    master_csv_path : str, optional
        Path to the unscored master csv. Default is MASTER_CSV_PATH.

    Returns
    -------
    pd.DataFrame
        The scored master dataframe (also saved to SCORED_MASTER_CSV_PATH).
    """
    scored_path = generate_sentiment_scores(master_csv_path, output_path=SCORED_MASTER_CSV_PATH)
    return generate_gap_scores(scored_path)