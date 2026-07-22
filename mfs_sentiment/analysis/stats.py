import pandas as pd
from scipy.stats import pearsonr


def app_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate aggregated statistics for each app across all reviews.
    
    Computes per-app metrics including review count, gap score statistics,
    mismatch category distributions, and correlation between normalized ratings
    and sentiment scores.
    
    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with all reviews (must contain columns: 'app_name',
        'rating', 'gap_score', 'mismatch_label', 'sentiment_score').
    
    Returns
    -------
    pd.DataFrame
        Summary statistics per app with columns:
        - app_name: app identifier
        - n_reviews: number of reviews for that app
        - mean_gap_score: average gap score
        - pct_inflated: percentage of reviews with 'Inflated' label
        - pct_deflated: percentage of reviews with 'Deflated' label
        - pct_consistent: percentage of reviews with 'Consistent' label
        - pearson_r: Pearson correlation coefficient
        - pearson_p_value: p-value for the correlation test
    
    Notes
    -----
    - Ratings are normalized to [-1, 1] range for correlation calculation
    - Rows with NaN sentiment_score or rating are excluded from correlation
    - Correlations test relationship between normalized rating and actual sentiment
    """
    summary_rows = []

    for app, group in df.groupby("app_name"):
        normalized_rating = (group["rating"] - 3) / 2

        valid = normalized_rating.notna() & group["sentiment_score"].notna()
        valid_ratings = normalized_rating[valid]
        valid_sentiment = group["sentiment_score"][valid]
        if valid.sum() < 2 or valid_ratings.nunique() < 2 or valid_sentiment.nunique() < 2:
            r_value, p_value = float("nan"), float("nan")
        else:
            r_value, p_value = pearsonr(valid_ratings, valid_sentiment)

        label_pct = group["mismatch_label"].value_counts(normalize=True) * 100

        summary_rows.append({
            "app_name": app,
            "n_reviews": len(group),
            "mean_gap_score": group["gap_score"].mean(),
            "pct_inflated": label_pct.get("Inflated", 0.0),
            "pct_deflated": label_pct.get("Deflated", 0.0),
            "pct_consistent": label_pct.get("Consistent", 0.0),
            "pearson_r": r_value,
            "pearson_p_value": p_value,
        })

    return pd.DataFrame(summary_rows)


def monthly_gap_trend(df: pd.DataFrame, warning: bool = False) -> pd.DataFrame:
    """
    Calculate monthly average gap scores per app for temporal trend analysis.
    
    Aggregates gap scores by app and calendar month to show how rating-sentiment
    mismatches evolve over time. Useful for time-series visualization (heatmaps,
    line charts).
    
    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with all reviews (must contain 'app_name',
        'review_date', and 'gap_score' columns).
    warning : bool, optional
        If True, print warning messages about unparseable dates. Default is False.
    
    Returns
    -------
    pd.DataFrame
        Monthly aggregated data with columns:
        - app_name: app identifier
        - year_month: calendar month as Period object (YYYY-MM format)
        - mean_gap_score: average gap score for that app in that month
    
    Notes
    -----
    - review_date is parsed with errors='coerce' (unparseable dates become NaT)
    - Unparseable rows are dropped before aggregation
    - Useful for detecting seasonal patterns or event-driven trends in mismatches
    - If warning=True, reports count of unparseable dates
    """
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    if warning:
        n_bad_dates = df["review_date"].isna().sum()
        if n_bad_dates:
            print(f"[WARNING] {n_bad_dates} rows have unparseable review_date")

    temp = df.dropna(subset=["review_date"]).copy()
    temp["year_month"] = temp["review_date"].dt.to_period("M")

    trend = (
        temp.groupby(["app_name", "year_month"])["gap_score"]
        .mean()
        .reset_index()
        .rename(columns={"gap_score": "mean_gap_score"})
    )
    return trend