import os
from pathlib import Path

from mfs_sentiment.analysis.lda import run_lda_on_subset
from mfs_sentiment.analysis.stats import app_level_summary, monthly_gap_trend
from mfs_sentiment.config import (
    DATASET_PATH_DICT,
    LDA_TOPIC_NUM,
    LDA_TOPWORD_NUM,
)
from mfs_sentiment.pipeline.master import load_master_csv
from mfs_sentiment.setup_nltk import ensure_nltk_data


def run_full_analysis(master_csv_path: Path|str, verbose: bool = True, lda_per_app: bool = True):
    """
    Execute complete analysis pipeline including app-level stats, trends, and LDA topic modeling.
    
    Loads master CSV, computes app-level statistics, monthly trends, and performs
    Latent Dirichlet Allocation (LDA) topic modeling. Can run LDA separately for each
    app and mismatch label, or pooled across all apps per label.
    
    Parameters
    ----------
    master_csv_path : str
        Path to the master CSV file with all processed reviews.
    verbose : bool, optional
        If True, prints all results (app-level stats, trends, topics) to console.
        If False, calculates results silently and returns them. Default is True.
    lda_per_app : bool, optional
        If True, runs LDA separately for each (app, label) combination (6 runs: 3 apps × 2 labels).
        If False, runs LDA once per label pooled across all apps (2 runs total).
        Default is True.
    
    Returns
    -------
    tuple
        If successful, returns (summary_df, trend_df, lda_results).
        
        If lda_per_app=True:
        - lda_results: dict with structure {app_name: {label: topics_list}}
        
        If lda_per_app=False:
        - lda_results: dict with structure {label: topics_list}
    
    Notes
    -----
    - Per-app LDA (lda_per_app=True) reveals app-specific themes for each mismatch type
    - Pooled LDA (lda_per_app=False) shows common patterns across all apps
    - Each LDA run uses its own fitted CountVectorizer and LDA model
    - Results are always saved to RESULT_DIR:
      - app_level_summary.csv: per-app statistics
      - monthly_gap_trend.csv: monthly aggregated gaps
    - verbose flag controls console output only; file output always occurs
    """
    df = load_master_csv(master_csv_path)
    if df is None:
        raise Exception("Incorrectly formatted master csv")

    result_dir = Path(master_csv_path).resolve().parent
    summary_df = app_level_summary(df)
    summary_df.to_csv(result_dir / "app_level_summary.csv", index=False)

    trend_df = monthly_gap_trend(df)
    trend_df.to_csv(result_dir / "monthly_gap_trend.csv", index=False)

    ensure_nltk_data()

    # Run LDA per app per label or per label
    if lda_per_app:
        lda_results = {}
        for app_name in DATASET_PATH_DICT.keys():
            lda_results[app_name] = {}
            app_df = df[df["app_name"] == app_name]
            
            for label in ["Inflated", "Deflated"]:
                _, topics = run_lda_on_subset(app_df, label, n_topics=LDA_TOPIC_NUM, n_top_words=LDA_TOPWORD_NUM)
                lda_results[app_name][label] = topics
    else:
        lda_results = {}
        for label in ["Inflated", "Deflated"]:
            _, topics = run_lda_on_subset(df, label, n_topics=LDA_TOPIC_NUM, n_top_words=LDA_TOPWORD_NUM)
            lda_results[label] = topics

    if verbose:
        print("=== App-Level Summary ===")
        print(summary_df)
        
        print("\n=== Monthly Gap Trends ===")
        print(trend_df)
        
        print("\n=== Topic Modeling Results ===")
        if lda_per_app:
            print("(Per App, Per Label)\n")
            for app_name in DATASET_PATH_DICT.keys():
                print(f"--- {app_name} ---")
                for label in ["Inflated", "Deflated"]:
                    print(f"  {label} Topics:")
                    topics = lda_results[app_name][label]
                    if topics:
                        for topic_info in topics:
                            print(f"    Topic {topic_info['topic']}: {', '.join(topic_info['top_words'])}")
                    else:
                        print(f"    [No topics generated - may indicate insufficient data]")
                print()
        else:
            print("(Pooled Across All Apps)\n")
            for label in ["Inflated", "Deflated"]:
                print(f"{label} Topics:")
                topics = lda_results[label]
                if topics:
                    for topic_info in topics:
                        print(f"  Topic {topic_info['topic']}: {', '.join(topic_info['top_words'])}")
                else:
                    print(f"  [No topics generated - may indicate insufficient data]")
                print()

    return summary_df, trend_df, lda_results