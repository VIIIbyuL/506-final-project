import pandas as pd
from transformers import pipeline

# Initialize the sentiment analysis pipeline with ProsusAI/finbert
model_name = "ProsusAI/finbert"
nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)

def get_sentiment_info(summary):
    # Process up to 512 tokens
    result = nlp(summary[:512])[0]
    # Print raw result for debugging
    print("Raw result:", result)
    # Mapping to handle both 'LABEL_x' and lowercase outputs
    mapping = {
        "LABEL_0": "Negative",
        "LABEL_1": "Neutral",
        "LABEL_2": "Positive",
        "NEGATIVE": "Negative",
        "NEUTRAL": "Neutral",
        "POSITIVE": "Positive"
    }
    # Use uppercase for mapping
    raw_label = result['label'].upper()
    label = mapping.get(raw_label, "Neutral")
    confidence = round(result['score'] * 100, 2)  # Convert to percent
    return label, confidence

def add_sentiment_from_summary_to_csv(input_file, output_file):
    df = pd.read_csv(input_file)
    if 'Summary' not in df.columns:
        raise ValueError("CSV must contain a 'Summary' column.")
    
    df['Summary'] = df['Summary'].fillna('').astype(str)
    
    sentiment_labels = []
    confidence_scores = []
    
    print("Processing summaries...\n")
    for index, row in df.iterrows():
        summary = row['Summary']
        label, confidence = get_sentiment_info(summary)
        print(f"Summary: {summary}")
        print(f"--> Sentiment: {label}, Confidence: {confidence}%\n")
        sentiment_labels.append(label)
        confidence_scores.append(confidence)
    
    df['finbert_sentiment_label'] = sentiment_labels
    df['finbert_confidence_percent'] = confidence_scores
    
    df.to_csv(output_file, index=False)
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    add_sentiment_from_summary_to_csv("data/company_news_v2.csv", "data/articles_with_finbert_sentiment.csv")
