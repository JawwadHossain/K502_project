import os
import math
import pandas as pd

def split_csv_into_chunks(path_to_csv: str, chunk_size: int) -> list[str]:
    """
    Splits a CSV into multiple smaller CSVs of fixed chunk size.

    Output naming format:
    appName_sample1_noOfRows.csv
    appName_sample2_noOfRows.csv
    ...
    """

    # Get filename without directory
    filename = os.path.basename(path_to_csv)
    name_part = filename.replace(".csv", "")

    # Expect format: appName_sample_noOfReviews
    parts = name_part.split("_")

    if len(parts) < 3:
        raise ValueError("Filename must be in format: appName_sample_noOfReviews.csv")

    app_name = parts[0]
    no_of_reviews = parts[-1]

    # Read CSV
    df = pd.read_csv(path_to_csv)

    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # ceiling division

    output_files = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size

        chunk_df = df.iloc[start:end]
        file_size = len(chunk_df)
        output_filename = f"{app_name}_sample{i+1}_{file_size}.csv"
        output_path = os.path.join(os.path.dirname(path_to_csv), output_filename)

        chunk_df.to_csv(output_path, index=False)

        output_files.append(output_path)

    return output_files



def compute_sample_counts(
    app_data: dict,
    total_reviews: int | None = None,
    ratio: float | None = None,
    verbose: bool = False
) -> dict:
    """
    Compute per-app sample counts based on total_reviews or ratio.
    Can be used standalone to estimate sample sizes before running the pipeline.

    Args:
        app_data      : dict  — {app_name: DataFrame} or {app_name: csv_path}
        total_reviews : int   — total samples across all apps (proportional per app)
        ratio         : float — fraction of each app's reviews to sample
        verbose       : bool  — print a summary table of counts

    Returns:
        dict — {app_name: sample_count}
    """
    if total_reviews is None and ratio is None:
        raise ValueError("Provide either total_reviews or ratio.")
    if total_reviews is not None and ratio is not None:
        raise ValueError("Provide only one of total_reviews or ratio, not both.")
    if ratio is not None and not (0 < ratio <= 1):
        raise ValueError("ratio must be between 0 (exclusive) and 1 (inclusive).")

    # Normalize: accept either DataFrames or csv paths
    dataframes = {}
    for app_name, value in app_data.items():
        if isinstance(value, str):
            dataframes[app_name] = pd.read_csv(value, encoding="utf-8-sig")
        elif isinstance(value, pd.DataFrame):
            dataframes[app_name] = value
        else:
            raise TypeError(f"Expected a file path or DataFrame for '{app_name}', got {type(value)}.")

    counts = {}

    if total_reviews is not None:
        total_all = sum(len(df) for df in dataframes.values())
        for app_name, df in dataframes.items():
            counts[app_name] = max(1, math.floor(len(df) / total_all * total_reviews))

        allocated = sum(counts.values())
        remainder = total_reviews - allocated
        if remainder > 0:
            fractional_losses = {
                app: (len(dataframes[app]) / total_all * total_reviews) - counts[app]
                for app in counts
            }

            fractional_items = sorted(
                fractional_losses.items(),
                key=lambda item: item[1],
                reverse=True
            )
            for app_name, _ in fractional_items[:remainder]:
                counts[app_name] += 1
    else:
        assert ratio is not None
        for app_name, df in dataframes.items():
            counts[app_name] = max(1, math.floor(len(df) * ratio))

    if verbose:
        total_all = sum(len(df) for df in dataframes.values())
        print(f"{'App':<15} {'Available':>10} {'Sampled':>10} {'Proportion':>12}")
        print("-" * 50)
        for app_name, df in dataframes.items():
            n = counts[app_name]
            print(f"{app_name:<15} {len(df):>10} {n:>10} {n/len(df):>11.1%}")
        print("-" * 50)
        print(f"{'TOTAL':<15} {total_all:>10} {sum(counts.values()):>10}")

    return counts


def sample_reviews(
    app_csv_map: dict,
    pipeline_fn,
    text_column: str = "review_text",
    total_reviews: int | None = None,
    ratio: float | None = None,
) -> None:

    dataframes = {}
    for app_name, csv_path in app_csv_map.items():
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in {csv_path}.")
        dataframes[app_name] = df[[text_column]].dropna().reset_index(drop=True)

    counts = compute_sample_counts(dataframes, total_reviews, ratio)

    for app_name, df in dataframes.items():
        n = min(counts[app_name], len(df))
        texts = df[text_column].sample(n=n, random_state=42).tolist()

        processed_df = pd.DataFrame([pipeline_fn(t, app_name) for t in texts])
        processed_df.insert(0, "app_name", app_name)
        processed_df["remarks"] = ""

        out_path = f"{app_name}_sample_{n}.csv"
        processed_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[{app_name}] Processed {n} reviews → {out_path}")