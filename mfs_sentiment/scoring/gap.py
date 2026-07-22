import pandas as pd
from pathlib import Path


def _compute_gap_score(sentiment_score: float, rating: int|float) -> float:
    """
    Calculate the gap score between normalized star rating and sentiment score.
    
    Measures the mismatch between user's star rating and the actual sentiment
    expressed in their review text. Normalizes both to [-1, 1] range and computes
    the difference normalized to [-1, 1].
    
    Parameters
    ----------
    sentiment_score : float
        VADER compound sentiment score in range [-1.0, 1.0].
    rating : int or float
        User's star rating in range [1, 5].
    
    Returns
    -------
    float
        Gap score in range [-1.0, 1.0].
        Positive = rating higher than sentiment (inflated rating)
        Negative = rating lower than sentiment (deflated rating)
        Near 0 = rating consistent with sentiment
    
    Notes
    -----
    - Ratings [1, 5] are mapped to [-1, 1] range via (rating - 3.0) / 2.0
    - Gap = (normalized_rating - sentiment_score) / 2.0
    """
    normalized_star = (rating - 3.0) / 2.0                  # maps [1, 5] to [-1, 1]
    gap_score = (normalized_star - sentiment_score) / 2.0   # Calculating gap score [-1, 1]   
    return gap_score


def _classify_gap_score(gap_score: float) -> str:
    """
    Classify gap score into a mismatch category.
    
    Categorizes the gap score into one of three labels based on thresholds:
    - Inflated: user gave a higher rating than sentiment suggests
    - Deflated: user gave a lower rating than sentiment suggests
    - Consistent: rating and sentiment are aligned
    
    Parameters
    ----------
    gap_score : float
        Gap score in range [-1.0, 1.0] (output from _compute_gap_score).
    
    Returns
    -------
    str
        One of: 'Inflated', 'Deflated', or 'Consistent'.
        - 'Inflated': gap_score > 0.25
        - 'Deflated': gap_score < -0.25
        - 'Consistent': gap_score in [-0.25, 0.25]
    
    Notes
    -----
    - Thresholds (±0.25) provide a tolerance for small mismatches
    """
    if gap_score > 0.25:
        return "Inflated"
    elif gap_score < -0.25:
        return "Deflated"
    else:
        return "Consistent"


def generate_gap_scores(input_path: Path|str) -> pd.DataFrame:
    """
    Compute gap scores and classify rating-sentiment mismatches for all reviews.
    
    Calculates gap score for each review by comparing normalized star rating with
    sentiment score, then classifies each review as Inflated/Deflated/Consistent.
    Adds 'gap_score' and 'mismatch_label' columns and saves updated data back to input file.
    
    Parameters
    ----------
    input_path : str
        Path to CSV file with reviews (must contain 'rating' and 'sentiment_score' columns).
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all original columns plus 'gap_score' and 'mismatch_label'.
        Data is also saved back to the input CSV file.
    
    Notes
    -----
    - Converts 'rating' to int and 'sentiment_score' to float before computation
    - Uses _compute_gap_score() to calculate gap for each row
    - Uses _classify_gap_score() to assign mismatch labels
    - Modifies and saves the input CSV file in-place
    """
    df = pd.read_csv(input_path)

    # ensuring correct dtypes before computation
    df["rating"] = df["rating"].astype(int)
    df["sentiment_score"] = df["sentiment_score"].astype(float)

    # Computing gap score
    df["gap_score"] = df.apply(
        lambda row: _compute_gap_score(row["sentiment_score"], row["rating"]),
        axis=1
    )

    # Assigning mismatch label (ratings w.r.t sentiment) > inflated/deflated/consistent
    df["mismatch_label"] = df["gap_score"].apply(_classify_gap_score)

    df.to_csv(input_path, index=False)
    return df