import pandas as pd

def build_alignment_dataset(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # drop rows without necessary values
    df = df.dropna(subset=["price_change_percent", "finbert_sentiment_label"])

    # binary label: 1 = aligned with price move, 0 = not aligned
    def label_alignment(sentiment, change):
        if sentiment == "Positive" and change > 0:
            return 1
        elif sentiment == "Negative" and change < 0:
            return 1
        elif sentiment == "Neutral":
            return 0 
        else:
            return 0

    df["alignment_label"] = df.apply(
        lambda row: label_alignment(row["finbert_sentiment_label"], row["price_change_percent"]),
        axis=1
    )

    df.to_csv(output_csv, index=False)
    print(f"Alignment-labeled dataset saved to: {output_csv}")
    print(f"Labeled rows: {len(df)}")


if __name__ == "__main__":
    build_alignment_dataset(
        "data/articles_with_price_change.csv",
        "data/articles_with_alignment_labels.csv"
    )
