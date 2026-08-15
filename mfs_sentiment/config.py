from pathlib import Path

# Find the project root by walking upward until requirements.txt is found
PROJECT_ROOT = Path.cwd().resolve()
for parent in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
    if (parent / "requirements.txt").exists():
        PROJECT_ROOT = parent
        break

# --- Directories ---
DATASET_DIR = PROJECT_ROOT / "data" / "raw"
FASTTEXT_MODEL_DIR = PROJECT_ROOT / "models" / "fasttext"
RESULT_DIR = PROJECT_ROOT / "results"
TRANSLATED_CSV_DIR = RESULT_DIR / "translated"
CLEANED_CSV_DIR = PROJECT_ROOT / "data" / "interim" / "cleaned"

FASTTEXT_MODEL_NAME = "lid.176.bin"                            # pretrained FastText language identification model
FASTTEXT_MODEL_PATH = FASTTEXT_MODEL_DIR / FASTTEXT_MODEL_NAME


def get_run_paths(only_english: bool = False) -> dict:
    """
    Return the output directories and CSV paths for a run mode
    """
    mode_name = "only_english" if only_english else "bangla_english"
    result_dir = PROJECT_ROOT / "results" / mode_name
    translated_dir = result_dir / "translated"
    cleaned_dir = PROJECT_ROOT / "data" / "interim" / "cleaned" / mode_name
    master_csv_path = result_dir / "master_bd_reviews.csv"
    scored_master_csv_path = result_dir / "master_bd_reviews_scored.csv"

    return {
        "mode_name": mode_name,
        "result_dir": result_dir,
        "translated_dir": translated_dir,
        "cleaned_dir": cleaned_dir,
        "master_csv_path": master_csv_path,
        "scored_master_csv_path": scored_master_csv_path,
    }

# Raw per-app review csv locations
DATASET_PATH_DICT = {
    "bKash":  DATASET_DIR / "bkash_bd_reviews.csv",
    "Nagad":  DATASET_DIR / "nagad_bd_reviews.csv",
    "Rocket": DATASET_DIR / "rocket_bd_reviews.csv",
}

# Combined cleaned-but-unscored master csv, built from all apps' cleaned reviews
MASTER_CSV_PATH = RESULT_DIR / "master_bd_reviews.csv"
# Master csv after sentiment + gap scoring has been applied
SCORED_MASTER_CSV_PATH = RESULT_DIR / "master_bd_reviews_scored.csv"

REQUIRED_RAW_CSV_COLUMNS = ['app_name', 'review_id', 'review_date', 'rating', 'review_text', 'thumbs_up_count', 'app_version']
REQUIRED_MASTER_CSV_COLUMNS = ['app_name', 'review_id', 'review_date', 'rating', 'review_text', 'review_text_clean', 'thumbs_up_count', 'app_version']
REQUIRED_SCORED_MASTER_CSV_COLUMNS = REQUIRED_MASTER_CSV_COLUMNS + ['sentiment_score', 'gap_score', 'mismatch_label']

# Domain-specific noise words to exclude from LDA topic modeling, on top of standard English stopwords
STOPWORDS_FOR_LDA = {"app", "bkash", "nagad", "rocket", "phone"}

LDA_MIN_THRESHOLD = 3   # min_df for CountVectorizer: ignore tokens appearing in fewer than this many reviews
LDA_TOPIC_NUM = 5       # number of topics to extract per LDA run
LDA_TOPWORD_NUM = 10    # number of top words to report per topic


def ensure_dirs(only_english: bool = False):
    """
    Create the appropriate output folders for either run mode
    """
    run_paths = get_run_paths(only_english=only_english)
    run_paths["result_dir"].mkdir(parents=True, exist_ok=True)
    run_paths["translated_dir"].mkdir(parents=True, exist_ok=True)
    FASTTEXT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    run_paths["cleaned_dir"].mkdir(parents=True, exist_ok=True)

    # Keep the legacy top-level folders for compatibility when not routing by mode.
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSLATED_CSV_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_CSV_DIR.mkdir(parents=True, exist_ok=True)