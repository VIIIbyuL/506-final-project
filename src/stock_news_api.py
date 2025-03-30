import finnhub
import requests
import pandas as pd
import datetime
from bs4 import BeautifulSoup
import time

# Finnhub API Key
FINNHUB_API_KEY = ""

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# List of company stock symbols
COMPANIES = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Boeing": "BA",
    "Vale": "VALE",
    "Amazon": "AMZN"
}

# Date range (last 2 years)
START_DATE = "2023-03-01"
END_DATE = "2025-03-01"

def fetch_news(ticker):
    """Fetches news articles for a specific company from Finnhub API."""
    try:
        news = finnhub_client.company_news(ticker, _from=START_DATE, to=END_DATE)
        return news if news else []
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

def scrape_article_content(url):
    """Scrapes the article content from the given URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract article content (common tags: p, article, div)
            paragraphs = soup.find_all("p")
            article_text = "\n".join([p.get_text() for p in paragraphs])

            return article_text if article_text else "Content not found"
        else:
            return "Failed to fetch article"
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "Scraping error"

def process_articles(articles, company_name):
    """Extract relevant information and scrape full content."""
    processed_data = []
    
    for article in articles:
        try:
            title = article.get("headline", "")
            link = article.get("url", "")
            source = article.get("source", "")
            timestamp = article.get("datetime", "")
            summary = article.get("summary", "")

            # Convert timestamp to a readable date format
            date = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp else "Unknown"

            # Scrape article content
            content = scrape_article_content(link)

            processed_data.append({
                "Company": company_name,
                "Date": date,
                "Title": title,
                "Source": source,
                "URL": link,
                "Summary": summary,
                "Content": content
            })

        except Exception as e:
            print(f"Error processing article: {e}")

    return processed_data

def save_to_csv(data, filename="company_news_v2.csv"):
    """Save the processed data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(data)} articles to {filename}")

if __name__ == "__main__":
    all_articles = []

    for company, ticker in COMPANIES.items():
        print(f"Fetching news for {company} ({ticker})...")
        articles = fetch_news(ticker)
        processed_articles = process_articles(articles, company)
        all_articles.extend(processed_articles)

    if all_articles:
        save_to_csv(all_articles)
    else:
        print("No articles were retrieved.")