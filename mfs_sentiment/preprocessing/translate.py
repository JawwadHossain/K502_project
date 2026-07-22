import time
from deep_translator import GoogleTranslator
from mfs_sentiment.preprocessing.text import is_banglish


def translate_to_english(text: str, source_lang: str, confidence: float, only_english: bool = False) -> str:
    """Translate non-English input to English, unless translation isn't needed.

    Parameters
    ----------
    text : str
        Text to translate.
    source_lang : str
        Language code detected for `text` (e.g. 'en').
    confidence : float
        Confidence score for the language detection, in [0, 1].
    only_english : bool, optional
        If True, skip translation entirely: rows that aren't English will be
        dropped downstream by `build_standarized_csv`, and English rows don't
        need translating. Default is False (translate as usual).

    Returns
    -------
    str
        The English text, or the original text if translation was skipped
        (either because it's already confidently English, or only_english=True).
    """
    if not text.strip():
        return text

    if only_english:
        # Non-English rows get filtered out later; English rows need no translation.
        return text

    is_confident_english = (
        source_lang == "en"
        and text.isascii()
        and confidence > 0.90
        and not is_banglish(text)
    )
    if is_confident_english:
        return text

    try:
        result = GoogleTranslator(source="auto", target="en").translate(text)
        time.sleep(0.3)
        return result
    except Exception as e:
        print(f"Translation error: {e}")
        return text