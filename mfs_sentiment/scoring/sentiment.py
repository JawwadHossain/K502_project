import os
import pandas as pd
from pathlib import Path
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def generate_sentiment_scores(input_path: Path|str, output_path: Path|str|None = None) -> Path|str:
    """
    Calculate sentiment scores for review texts using the VADER sentiment analyzer.

    Parameters
    ----------
    input_path : str
        Path to input CSV file (must contain 'review_text_clean' column).
    output_path : str, optional
        Where to write the scored csv. If None, saves alongside input_path
        with a '_scored' suffix. Default is None.

    Returns
    -------
    str
        Path to the output CSV file with sentiment scores.
    """
    df = pd.read_csv(input_path)

    analyzer = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["review_text_clean"].apply(
        lambda text: analyzer.polarity_scores(str(text))["compound"]
    )

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_scored{ext}"
    df.to_csv(output_path, index=False)
    return output_path