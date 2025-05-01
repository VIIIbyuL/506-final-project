"""
Runs the full end-to-end pipeline:
1. (Optional) Scrapes news articles from Finnhub API
2. Performs FinBERT sentiment analysis
3. Fetches stock price movement using yFinance
4. Labels alignment between sentiment and price
5. Trains & evaluates classification model
6. Generates visualizations and trend forecasts
"""


from sentiment_analysis import run_sentiment_pipeline
from price_fetching import add_price_change_to_csv
from generate_labels import build_alignment_dataset
from model import main as run_all_evaluations
from stock_news_api import run_news_scraper

def main():
    # For speed, we have already ran the scraper into the data/company_news_v2.csv file. 
    # If you want to re-run the scraper, uncomment the following lines just be aware that
    # it will take a while to run.

    # print(" Scraping News Articles...")
    # run_news_scraper("data/company_news_v2.csv")

    print(" Starting Sentiment Analysis...")
    run_sentiment_pipeline("data/company_news_v2.csv", "data/articles_with_finbert_sentiment.csv")

    print(" Fetching Stock Price Change...")
    add_price_change_to_csv("data/articles_with_finbert_sentiment.csv", "data/articles_with_price_change.csv")

    print(" Generating Alignment Labels...")
    build_alignment_dataset("data/articles_with_price_change.csv", "data/articles_with_alignment_labels.csv")

    print(" Running Model Evaluations + Visualizations...")
    run_all_evaluations()

if __name__ == "__main__":
    main()
