_NLTK_READY = False


def ensure_nltk_data() -> None:
    """Download NLTK resources required for LDA preprocessing."""
    global _NLTK_READY
    if _NLTK_READY:
        return

    import nltk

    for resource in ("punkt", "punkt_tab", "stopwords", "wordnet"):
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(resource, quiet=True)

    _NLTK_READY = True
