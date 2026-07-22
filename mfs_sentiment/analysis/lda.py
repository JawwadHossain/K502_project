import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from mfs_sentiment.config import LDA_MIN_THRESHOLD, STOPWORDS_FOR_LDA
from mfs_sentiment.setup_nltk import ensure_nltk_data

_lemmatizer = None
_custom_stopwords = None


def _ensure_lda_resources() -> None:
    global _lemmatizer, _custom_stopwords
    ensure_nltk_data()
    if _custom_stopwords is None or _lemmatizer is None:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        _lemmatizer = WordNetLemmatizer()
        _custom_stopwords = set(stopwords.words("english")) | STOPWORDS_FOR_LDA


def _preprocess_for_lda(text: str) -> str:
    """
    Preprocess review text for Latent Dirichlet Allocation (LDA) topic modeling.
    
    Performs multi-step text cleaning: tokenization, alphabetic filtering,
    stopword removal, and lemmatization to prepare text for LDA analysis.
    
    Parameters
    ----------
    text : str
        Raw review text to preprocess.
    
    Returns
    -------
    str
        Preprocessed text as space-separated tokens ready for LDA.
        Tokens are lowercase, lemmatized, and free of stopwords and punctuation.
    
    Notes
    -----
    - Converts text to lowercase during tokenization
    - Removes non-alphabetic characters (punctuation, numbers, symbols)
    - Filters out English stopwords plus custom domain stopwords (STOPWORDS_FOR_LDA)
    - Lemmatizes tokens to root form (e.g., 'running' → 'run')
    - Requires NLTK tokenizer, lemmatizer, and stopwords to be downloaded
    - Use sparingly; preprocessing can be time-intensive for large datasets
    """
    from nltk.tokenize import word_tokenize

    _ensure_lda_resources()
    tokens = word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha()]                 # Discarding punctuation/numbers
    tokens = [t for t in tokens if t not in _custom_stopwords]  # Removing stopwords
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]         # Converting words to root form: running -> run
    return " ".join(tokens)


def run_lda_on_subset(df: pd.DataFrame, label: str, n_topics: int = 5, n_top_words: int = 10):
    """
    Perform Latent Dirichlet Allocation (LDA) topic modeling on a mismatch subset.
    
    Filters reviews by mismatch label (Inflated/Deflated/Consistent), preprocesses
    text, builds a CountVectorizer and LDA model, and extracts top words per topic.
    Each subset gets its own fitted vectorizer and LDA model (not shared).
    
    Parameters
    ----------
    df : pd.DataFrame
        Master dataframe with all reviews (must contain 'review_text_clean'
        and 'mismatch_label' columns).
    label : str
        Mismatch label to filter on: 'Inflated', 'Deflated', or 'Consistent'.
    n_topics : int, optional
        Number of topics for LDA model. Default is 5.
    n_top_words : int, optional
        Number of top words to extract per topic. Default is 10.
    
    Returns
    -------
    tuple(lda_model, topics) or (None, None)
        lda_model : sklearn.decomposition.LatentDirichletAllocation or None
            Fitted LDA model. None if subset is empty.
        topics : list of dict or None
            List of dictionaries with format:
            [{'topic': int, 'top_words': [str, ...]}, ...]
            None if subset is empty.
    
    Notes
    -----
    - Preprocesses text using preprocess_for_lda() before vectorization
    - CountVectorizer: max_df=0.9 (ignore tokens in >90% docs),
      min_df=LDA_MIN_THRESHOLD (ignore tokens in fewer than that many docs)
    - LDA uses random_state=42 for reproducibility
    - Separate vectorizer per subset ensures vocabulary reflects mismatch-specific themes
    - Prints warning if label not found in DataFrame
    """
    subset = df[df["mismatch_label"] == label].copy()

    if subset.empty:
        print(f"[WARNING] No rows found for label '{label}'")
        return None, None

    if len(subset) < LDA_MIN_THRESHOLD:
        print(
            f"[WARNING] Too few rows for LDA on label '{label}' "
            f"({len(subset)} < {LDA_MIN_THRESHOLD})"
        )
        return None, None

    subset["lda_ready_text"] = subset["review_text_clean"].apply(_preprocess_for_lda)

    vectorizer = CountVectorizer(max_df=0.9, min_df=LDA_MIN_THRESHOLD)
    doc_term_matrix = vectorizer.fit_transform(subset["lda_ready_text"])

    lda_model = LatentDirichletAllocation(
        n_components=n_topics, random_state=42
    )
    lda_model.fit(doc_term_matrix)

    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [float(topic[i]) for i in top_indices]
        topics.append({"topic": topic_idx, "top_words": top_words, "top_weights": top_weights,})

    return lda_model, topics