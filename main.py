import argparse

from mfs_sentiment.config import (
    REQUIRED_MASTER_CSV_COLUMNS,
    ensure_dirs,
    get_run_paths,
)
from mfs_sentiment.pipeline.master import build_master_csv, load_master_csv, score_master_csv
from mfs_sentiment.pipeline.run import run_full_analysis
from mfs_sentiment.viz.plots import (
    plot_lda_wordclouds,
    plot_mismatch_composition,
    plot_monthly_gap_heatmap,
    plot_monthly_gap_trend,
    plot_rating_vs_sentiment,
    plot_rating_vs_sentiment_by_app,
)


def main() -> None:
    """Run the full sentiment-analysis pipeline from the command line.

    Parses CLI flags, rebuilds or loads the master CSV, scores reviews,
    executes analysis, and optionally renders visualizations.
    """
    parser = argparse.ArgumentParser(description="Run MFS sentiment analysis pipeline.")
    parser.add_argument("--only-english", action="store_true")
    parser.add_argument("--rebuild", action="store_true", help="Ignore cached CSVs")
    parser.add_argument("--skip-viz", action="store_true")
    args = parser.parse_args()

    ensure_dirs(only_english=args.only_english)
    run_paths = get_run_paths(only_english=args.only_english)

    # Step 1: build or load the unscored master CSV
    master_rebuilt = False
    if args.rebuild or not run_paths["master_csv_path"].exists():
        master_df = build_master_csv(only_english=args.only_english)
        master_rebuilt = True
    else:
        master_df = load_master_csv(
            run_paths["master_csv_path"],
            required_columns=REQUIRED_MASTER_CSV_COLUMNS,
        )

    if master_df is None:
        raise ValueError("Failed to load or build master CSV.")

    # Step 2: score or load the scored master CSV
    if args.rebuild or master_rebuilt or not run_paths["scored_master_csv_path"].exists():
        scored_df = score_master_csv(run_paths["master_csv_path"], only_english=args.only_english)
    else:
        scored_df = load_master_csv(run_paths["scored_master_csv_path"])

    if scored_df is None:
        raise ValueError("Failed to load or score master CSV.")

    # Step 3: analysis
    summary_df, trend_df, lda_results = run_full_analysis(
        run_paths["scored_master_csv_path"],
        verbose=True,
        lda_per_app=True,
    )

    # Step 4: visualizations
    if not args.skip_viz:
        plot_rating_vs_sentiment(scored_df)
        plot_rating_vs_sentiment_by_app(scored_df)
        plot_mismatch_composition(summary_df)
        plot_monthly_gap_heatmap(trend_df)
        plot_lda_wordclouds(lda_results)
        plot_monthly_gap_trend(trend_df)


if __name__ == "__main__":
    main()
