import pandas as pd

def test_alignment_label_column_exists():
    df = pd.read_csv("data/articles_with_alignment_labels.csv")
    assert "alignment_label" in df.columns

def test_no_missing_sentiment():
    df = pd.read_csv("data/articles_with_alignment_labels.csv")
    assert df["finbert_sentiment_label"].notnull().all()

def test_price_change_is_number():
    df = pd.read_csv("data/articles_with_alignment_labels.csv")
    assert pd.to_numeric(df["price_change_percent"], errors="coerce").notnull().all()
