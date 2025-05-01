import finnhub
import requests
import pandas as pd
import datetime
from bs4 import BeautifulSoup
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import time

# Finnhub API Key
FINNHUB_API_KEY = "cvi916pr01qks9q91q00cvi916pr01qks9q91q0g"

# Initialize Finnhub client
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# List of company stock symbols
COMPANIES = {
    "Apple": "AAPL",
    "Tesla": "TSLA",
    "Amazon": "AMZN",
}

# Date range
START_DATE = "2024-05-01"
END_DATE = "2025-05-01"
WINDOW_SIZE_DAYS = 7

# Throttled API call setup
API_CALL_INTERVAL = 0.1  # seconds
api_call_queue = queue.Queue()
fetched_results = {}

# API Worker
def api_worker():
    while True:
        task = api_call_queue.get()
        if task is None:
            break

        ticker, start_date, end_date, result_key = task
        try:
            print(f"Calling API: {ticker} {start_date} to {end_date}")
            result = finnhub_client.company_news(ticker, _from=start_date, to=end_date)
            fetched_results[result_key] = result if result else []
        except Exception as e:
            print(f"API error: {e}")
            fetched_results[result_key] = []
        finally:
            time.sleep(API_CALL_INTERVAL)
            api_call_queue.task_done()

# Start the API worker thread
api_thread = threading.Thread(target=api_worker, daemon=True)
api_thread.start()

def sliding_window_fetch(ticker):
    current_start_date = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    final_end_date = datetime.datetime.strptime(END_DATE, '%Y-%m-%d')

    result_keys = []
    while current_start_date < final_end_date:
        window_end_date = current_start_date + datetime.timedelta(days=WINDOW_SIZE_DAYS)
        if window_end_date > final_end_date:
            window_end_date = final_end_date

        start_str = current_start_date.strftime('%Y-%m-%d')
        end_str = window_end_date.strftime('%Y-%m-%d')
        result_key = f"{ticker}_{start_str}_{end_str}"
        result_keys.append(result_key)

        api_call_queue.put((ticker, start_str, end_str, result_key))
        current_start_date = window_end_date

    # Wait for all API calls to complete
    api_call_queue.join()

    all_articles = []
    for key in result_keys:
        all_articles.extend(fetched_results.get(key, []))
    return all_articles

def scrape_article_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=5)
        print('scraping', datetime.datetime.now())

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text() for p in paragraphs])
            cleaned_text = re.sub(r'\s+', ' ', text).strip()
            return cleaned_text or "Content not found"
        else:
            return "Failed to fetch article"
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return "Scraping error"

def process_article(article, company_name):
    try:
        title = article.get("headline", "")
        link = article.get("url", "")
        source = article.get("source", "")
        timestamp = article.get("datetime", "")
        summary = article.get("summary", "")
        date = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d') if timestamp else "Unknown"
        content = scrape_article_content(link)
        print('processing', datetime.datetime.now())

        return {
            "Company": company_name,
            "Date": date,
            "Title": title,
            "Source": source,
            "URL": link,
            "Summary": summary,
            "Content": content
        }
    except Exception as e:
        print(f"Error processing article: {e}")
        return None

def save_to_csv(data, filename="company_news_v2.csv"):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved {len(data)} articles to {filename}")

if __name__ == "__main__":
    all_articles = []

    for company, ticker in COMPANIES.items():
        print(f"Fetching news for {company} ({ticker})...")
        articles = sliding_window_fetch(ticker)

        with ThreadPoolExecutor(max_workers=10) as process_pool:
            process_futures = [process_pool.submit(process_article, article, company) for article in articles]

            for p_future in as_completed(process_futures):
                result = p_future.result()
                if result:
                    all_articles.append(result)

    if all_articles:
        save_to_csv(all_articles, 'more_news.csv')
    else:
        print("No articles were retrieved.")

    # Stop the API worker thread cleanly
    api_call_queue.put(None)
    api_thread.join()