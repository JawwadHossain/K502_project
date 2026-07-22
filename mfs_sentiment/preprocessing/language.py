from langdetect import LangDetectException, detect
from mfs_sentiment.models import get_fasttext_model


def detect_language(text: str) -> tuple:
    """Detect the language of a review using FastText with a langdetect fallback."""
    _FT_MODEL = get_fasttext_model()
    try:
        label, confidence = _FT_MODEL.predict(text.replace("\n", " "), k=1)
        lang = label[0].replace("__label__", "")  # type: ignore
        return lang, float(confidence[0])
    except Exception:
        pass

    try:
        return detect(text), 0.0
    except LangDetectException:
        return "unknown", 0.0