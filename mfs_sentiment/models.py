import urllib.request
import fasttext

from mfs_sentiment.config import FASTTEXT_MODEL_DIR, FASTTEXT_MODEL_NAME, FASTTEXT_MODEL_PATH


def ensure_fasttext_model() -> None:
    """Download the FastText language-id model if it is not already present."""
    FASTTEXT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not FASTTEXT_MODEL_PATH.exists():
        print("Downloading FastText language model...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/" + FASTTEXT_MODEL_NAME,
            FASTTEXT_MODEL_PATH,
        )
        print("Download complete.")


_FT_MODEL = None

def get_fasttext_model():
    """Load and cache the FastText language-identification model."""
    global _FT_MODEL
    if _FT_MODEL is None:
        ensure_fasttext_model()
        _FT_MODEL = fasttext.load_model(str(FASTTEXT_MODEL_PATH))
    return _FT_MODEL
