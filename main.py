import argparse

from mfs_sentiment.config import (
    MASTER_CSV_PATH,
    REQUIRED_MASTER_CSV_COLUMNS,
    SCORED_MASTER_CSV_PATH,
    ensure_dirs,
)
from mfs_sentiment.pipeline.master import build_master_csv, load_master_csv, score_master_csv
from mfs_sentiment.pipeline.run import run_full_analysis
from mfs_sentiment.viz.plots import (
    plot_lda_wordclouds,
    plot_mismatch_composition,
    plot_monthly_gap_heatmap,
    plot_monthly_gap_trend,
    plot_rating_vs_sentiment_by_app,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MFS sentiment analysis pipeline.")
    parser.add_argument("--only-english", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Ignore cached CSVs")
    parser.add_argument("--skip-viz", action="store_true")
    args = parser.parse_args()

    ensure_dirs()

    # Step 1: build or load the unscored master CSV
    master_rebuilt = False
    if args.rebuild or not MASTER_CSV_PATH.exists():
        master_df = build_master_csv(only_english=args.only_english)
        master_rebuilt = True
    else:
        master_df = load_master_csv(
            MASTER_CSV_PATH,
            required_columns=REQUIRED_MASTER_CSV_COLUMNS,
        )

    if master_df is None:
        raise ValueError("Failed to load or build master CSV.")

    # Step 2: score or load the scored master CSV
    if args.rebuild or master_rebuilt or not SCORED_MASTER_CSV_PATH.exists():
        scored_df = score_master_csv(MASTER_CSV_PATH)
    else:
        scored_df = load_master_csv(SCORED_MASTER_CSV_PATH)

    if scored_df is None:
        raise ValueError("Failed to load or score master CSV.")

    # Step 3: analysis
    summary_df, trend_df, lda_results = run_full_analysis(
        SCORED_MASTER_CSV_PATH,
        verbose=True,
        lda_per_app=True,
    )

    # Step 4: visualizations
    if not args.skip_viz:
        plot_rating_vs_sentiment_by_app(scored_df)
        plot_mismatch_composition(summary_df)
        plot_monthly_gap_heatmap(trend_df)
        plot_lda_wordclouds(lda_results)
        plot_monthly_gap_trend(trend_df)


if __name__ == "__main__":
    main()
