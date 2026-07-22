import os
from pathlib import Path

import pandas as pd

from mfs_sentiment.config import (
    CLEANED_CSV_DIR,
    REQUIRED_RAW_CSV_COLUMNS,
    TRANSLATED_CSV_DIR,
)
from mfs_sentiment.dictionaries import (
    APP_SPECIFIC_TRANSLATION_DICT,
    POST_TRANSLATION_DICT,
    PRE_TRANSLATION_DICT,
)
from mfs_sentiment.preprocessing.language import detect_language
from mfs_sentiment.preprocessing.text import (
    apply_custom_dict,
    clean_text,
    convert_emojis,
)
from mfs_sentiment.preprocessing.translate import translate_to_english


def _run_standardization_steps(text: str, app_name: str, only_english: bool = False) -> dict:
    """Normalize a single review through the full standardization pipeline.

    Applies app-specific translations, emoji normalization, language detection,
    translation to English (unless only_english=True), cleaning, and a final
    post-processing pass, returning the standardized text plus intermediate
    artifacts used for inspection.

    Parameters
    ----------
    text : str
        Raw review text.
    app_name : str
        App the review belongs to; selects the app-specific translation dict.
    only_english : bool, optional
        If True, skip the actual translation call (see translate_to_english).
        Default is False.

    Returns
    -------
    dict
        Keys: original_text, pre_translated_text, emoji_converted, detected_lang,
        confidence, translated_text, cleaned_text, post_translated (final text).
    """
    text = text.lower()
    pre_translated = apply_custom_dict(text, PRE_TRANSLATION_DICT | APP_SPECIFIC_TRANSLATION_DICT[app_name])
    emoji_converted = convert_emojis(pre_translated)
    detected_lang, confidence = detect_language(emoji_converted)
    translated = translate_to_english(emoji_converted, detected_lang, confidence, only_english=only_english)
    cleaned = clean_text(translated)
    post_translated = apply_custom_dict(cleaned, POST_TRANSLATION_DICT)

    return {
        "original_text": text,
        "pre_translated_text": pre_translated,
        "emoji_converted": emoji_converted,
        "detected_lang": detected_lang,
        "confidence": round(float(confidence), 3),
        "translated_text": translated,
        "cleaned_text": cleaned,
        "post_translated": post_translated,
    }


def build_standarized_csv(input_path: str|Path, only_english: bool = False) -> Path:
    """Clean, standardize, and save review data from a raw CSV file.

    Reads the raw reviews, keeps required columns, fills missing app versions,
    runs the normalization pipeline row-by-row, and writes both the translated
    intermediate results and the cleaned dataset to CLEANED_CSV_DIR (a temp dir).

    Parameters
    ----------
    input_path : str
        Path to a raw per-app review csv (see DATASET_PATH_DICT).
    only_english : bool, optional
        If True, translation is skipped and only reviews detected as English
        are kept in the cleaned output. If False (default), all reviews are
        kept, with non-English ones translated to English.

    Returns
    -------
    str
        Path to the cleaned csv, written under CLEANED_CSV_DIR.
    """
    df = pd.read_csv(input_path)
    df = df[REQUIRED_RAW_CSV_COLUMNS].copy()
    df = df.dropna(subset=['review_text'])
    df['app_version'] = df['app_version'].fillna('unknown')

    step_results = df.apply(
        lambda row: _run_standardization_steps(row['review_text'], row['app_name'], only_english=only_english),
        axis=1,
    )
    steps_df = pd.DataFrame(step_results.tolist())
    steps_df.insert(0, 'review_id', df['review_id'].values.tolist())

    if only_english:
        # Rows that weren't translated (non-English) are dropped here.
        english_mask = (steps_df['detected_lang'] == 'en').values
        df = df[english_mask].reset_index(drop=True)
        steps_df = steps_df[english_mask].reset_index(drop=True)

    base, ext = os.path.splitext(os.path.basename(input_path))
    steps_df.to_csv(TRANSLATED_CSV_DIR / f"{base}_translated{ext}", index=False)

    idx = df.columns.get_loc('review_text')
    df.insert(idx + 1, 'review_text_clean', steps_df['post_translated'].values)  # type: ignore

    output_path = CLEANED_CSV_DIR / f"{base}_cleaned{ext}"
    df.to_csv(output_path, index=False)
    return output_path