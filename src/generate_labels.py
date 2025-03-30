"""
this script generates a dataset with alignment labels for articles
based on their sentiment and price change. If the sentiment is positive
and the price change is positive, or if the sentiment is negative and
the price change is negative, the label is 1 (aligned). If the sentiment
is neutral, the label is 0. If the sentiment is negative and the price
change is positive, or if the sentiment is positive and the price change
is negative, the label is 0 (not aligned).
"""

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

    # apply the labeling function to each row
    df["alignment_label"] = df.apply(
        lambda row: label_alignment(row["finbert_sentiment_label"], row["price_change_percent"]),
        axis=1
    )

    # save the labeled dataset
    df.to_csv(output_csv, index=False)
    print(f"Alignment-labeled dataset saved to: {output_csv}")
    print(f"Labeled rows: {len(df)}")

# main function to run the script
if __name__ == "__main__":
    build_alignment_dataset(
        "data/articles_with_price_change.csv",
        "data/articles_with_alignment_labels.csv"
    )
