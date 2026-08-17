import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from wordcloud import WordCloud

from mfs_sentiment.config import RESULT_DIR


def plot_rating_vs_sentiment(df: pd.DataFrame, save_path: str = os.path.join(RESULT_DIR, "scatter_rating_vs_sentiment.png")) -> Axes:
    """Scatter plot of normalized star rating vs. VADER sentiment score, per app.

    Parameters
    ----------
    df : pd.DataFrame
        Scored master dataframe (must contain 'app_name', 'rating', 'sentiment_score').
    save_path : str, optional
        Where to save the figure. Pass None to skip saving.
        Default is RESULT_DIR/scatter_rating_vs_sentiment.png.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for app_name, group in df.groupby("app_name"):
        normalized_rating = (group["rating"] - 3) / 2
        ax.scatter(normalized_rating, group["sentiment_score"], alpha=0.3, s=15, label=app_name)
    ax.plot([-1, 1], [-1, 1], linestyle="--", color="gray", linewidth=1, label="Perfect agreement")
    ax.set_xlabel("Normalized Star Rating")
    ax.set_ylabel("VADER Sentiment Score")
    ax.set_title("Rating vs. Sentiment by App")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax


def plot_rating_vs_sentiment_by_app(
    df: pd.DataFrame,
    apps=None,
    save_path: str = os.path.join(RESULT_DIR, "scatter_rating_vs_sentiment_by_app.png"),
) -> np.ndarray:
    """Create one scatter plot per app in separate subplots.

    Parameters
    ----------
    df : pd.DataFrame
        Scored master dataframe (must contain 'app_name', 'rating', 'sentiment_score').
    apps : list, optional
        App names to plot. If None, the first three apps found in the dataframe are used.
    save_path : str, optional
        Where to save the figure. Pass None to skip saving.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
    """
    if apps is None:
        apps = [app for app in ["bKash", "Nagad", "Rocket"] if app in set(df["app_name"].astype(str))]
        if not apps:
            apps = list(df["app_name"].dropna().astype(str).unique())[:3]

    selected_apps = [app for app in apps if app in set(df["app_name"].astype(str))]
    if not selected_apps:
        raise ValueError("No matching app names found in the dataframe.")

    fig, axes = plt.subplots(1, len(selected_apps), figsize=(5 * len(selected_apps), 5), squeeze=False)
    for ax, app_name in zip(axes.flat, selected_apps):
        group = df[df["app_name"] == app_name]
        normalized_rating = (group["rating"] - 3) / 2
        ax.scatter(normalized_rating, group["sentiment_score"], alpha=0.3, s=15)
        ax.plot([-1, 1], [-1, 1], linestyle="--", color="gray", linewidth=1, label="Perfect agreement")
        ax.set_xlabel("Normalized Star Rating")
        ax.set_ylabel("VADER Sentiment Score")
        ax.set_title(app_name)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    fig.suptitle("Rating vs. Sentiment by App")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save_path:
        plt.savefig(save_path, dpi=150)
    return axes


def plot_mismatch_composition(summary_df: pd.DataFrame, save_path: str = os.path.join(RESULT_DIR, "stacked_bar_mismatch.png")) -> Axes:
    """Stacked bar chart of Inflated/Deflated/Consistent review percentages per app.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of app_level_summary(); must contain 'app_name', 'pct_inflated',
        'pct_deflated', 'pct_consistent'.
    save_path : str, optional
        Where to save the figure. Pass None to skip saving.
        Default is RESULT_DIR/stacked_bar_mismatch.png.

    Returns
    -------
    matplotlib.axes.Axes
    """
    stacked_data = summary_df.set_index("app_name")[["pct_inflated", "pct_deflated", "pct_consistent"]]
    fig, ax = plt.subplots(figsize=(8, 6))
    stacked_data.plot(kind="bar", stacked=True, ax=ax, color=["#d9534f", "#5bc0de", "#5cb85c"])
    ax.set_ylabel("% of Reviews")
    ax.set_title("Rating-Sentiment Mismatch Distribution by App")
    ax.legend(title="Mismatch Type")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax


def plot_monthly_gap_heatmap(trend_df: pd.DataFrame, save_path: str = os.path.join(RESULT_DIR, "monthly_gap_heatmap.png")) -> Axes:
    """Heatmap of mean gap score by app (rows) and month (columns).

    Parameters
    ----------
    trend_df : pd.DataFrame
        Output of monthly_gap_trend(); must contain 'app_name', 'year_month', 'mean_gap_score'.
    save_path : str, optional
        Where to save the figure. Pass None to skip saving.
        Default is RESULT_DIR/monthly_gap_heatmap.png.

    Returns
    -------
    matplotlib.axes.Axes
    """
    heatmap_data = trend_df.pivot(index="app_name", columns="year_month", values="mean_gap_score")
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(heatmap_data, cmap="RdBu_r", center=0, cbar_kws={"label": "Mean Gap Score"}, ax=ax)
    ax.set_title("Monthly Mean Gap Score by App")
    ax.set_xlabel("Month")
    ax.set_ylabel("App")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax


def plot_lda_wordclouds(lda_results: dict, save_path: str = os.path.join(RESULT_DIR, "wordclouds.png")) -> np.ndarray:
    """Grid of word clouds, one per (app, mismatch label), from LDA topic weights.

    Parameters
    ----------
    lda_results : dict
        Per-app LDA results with structure {app_name: {label: topics_list}},
        as returned by run_full_analysis(lda_per_app=True).
    save_path : str, optional
        Where to save the figure. Pass None to skip saving.
        Default is RESULT_DIR/wordclouds.png.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
    """
    apps = list(lda_results.keys())
    fig, axes = plt.subplots(len(apps), 2, figsize=(12, 4 * len(apps)))
    if len(apps) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, app_name in enumerate(apps):
        for col_idx, label in enumerate(["Inflated", "Deflated"]):
            topics = lda_results[app_name].get(label)
            ax = axes[row_idx, col_idx]
            if not topics:
                ax.text(0.5, 0.5, "No topics generated", ha="center", va="center")
                ax.axis("off")
                continue

            word_freq = {}
            for topic_info in topics:
                for word, weight in zip(topic_info["top_words"], topic_info["top_weights"]):
                    word_freq[word] = word_freq.get(word, 0) + weight

            wc = WordCloud(width=500, height=350, background_color="white").generate_from_frequencies(word_freq)
            ax.imshow(wc, interpolation="bilinear")
            ax.set_title(f"{app_name} — {label}")
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return axes


def plot_monthly_gap_trend(trend_df: pd.DataFrame, save_path: str = os.path.join(RESULT_DIR, "monthly_gap_line_chart.png")) -> Axes:
    """Line chart of monthly mean gap score, one line per app.

    Parameters
    ----------
    trend_df : pd.DataFrame
        Output of monthly_gap_trend(); must contain 'app_name', 'year_month', 'mean_gap_score'.
    save_path : str, optional
        Where to save the figure. Pass None to skip saving.
        Default is RESULT_DIR/monthly_gap_line_chart.png.

    Returns
    -------
    matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    for app_name, group in trend_df.groupby("app_name"):
        group_sorted = group.sort_values("year_month")
        x_values = group_sorted["year_month"].dt.to_timestamp()
        ax.plot(x_values, group_sorted["mean_gap_score"], marker="o", label=app_name)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean Gap Score")
    ax.set_title("Monthly Mean Gap Score Trend by App")
    ax.legend(title="App")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return ax