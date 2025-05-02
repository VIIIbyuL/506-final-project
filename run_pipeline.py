"""
Runs the full end-to-end pipeline:
1. (Optional) Scrapes news articles from Finnhub API
2. Performs FinBERT sentiment analysis
3. Fetches stock price movement using yFinance
4. Labels alignment between sentiment and price
5. Trains & evaluates classification model
6. Generates visualizations and trend forecasts
"""


from src.sentiment_analysis import add_sentiment_from_summary_to_csv
from src.price_fetching import add_price_change_to_csv
from src.generate_labels import build_alignment_dataset
from src.model import main as run_all_evaluations


def main():
    # For speed, we have already ran the scraper into the data/more_news.csv file.

    # print(" Starting Sentiment Analysis...")
    # add_sentiment_from_summary_to_csv("data/more_news.csv", "data/articles_with_finbert_sentiment.csv")

    # print(" Fetching Stock Price Change...")
    # add_price_change_to_csv("data/articles_with_finbert_sentiment.csv", "data/articles_with_price_change.csv")

    # print(" Generating Alignment Labels...")
    # build_alignment_dataset("data/articles_with_price_change.csv", "data/articles_with_alignment_labels.csv")

    print(" Running Model Evaluations + Visualizations...")
    run_all_evaluations()

if __name__ == "__main__":
    main()
