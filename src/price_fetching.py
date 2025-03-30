"""
This script fetches the price change percentage of a stock from Yahoo Finance
for a given date and adds this information to a CSV file containing articles
with their corresponding companies and dates. The script uses the yfinance
library to fetch stock data and pandas for data manipulation.
"""

import pandas as pd
import yfinance as yf
import datetime

def get_price_change(ticker, date_str):
    dt = pd.to_datetime(date_str)

    # fetch up to 5 days of data to ensure at least 2 valid trading days
    data = yf.download(ticker, start=dt.strftime('%Y-%m-%d'),
                       end=(dt + datetime.timedelta(days=5)).strftime('%Y-%m-%d'),
                       progress=False)
    
    # keep only trading days on or after the given date
    data = data[data.index >= dt]

    if data.empty or len(data) < 2:
        return None

    # open from first valid trading day, Close from second
    open_price = data.iloc[0]['Open']
    close_price = data.iloc[1]['Close']
    change_percent = ((close_price - open_price) / open_price) * 100
    return round(float(change_percent), 2)

def add_price_change_to_csv(input_file, output_file):
    df = pd.read_csv(input_file)

    if 'Date' not in df.columns or 'Company' not in df.columns:
        raise ValueError("CSV must contain 'Date' and 'Company' columns.")

    company_to_ticker = {
        "Apple": "AAPL",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Meta": "META",
        "Tesla": "TSLA"
    }

    df["Ticker"] = df["Company"].map(company_to_ticker)

    price_changes = []
    for idx, row in df.iterrows():
        ticker = row["Ticker"]
        date_str = row["Date"]

        if pd.isna(ticker):
            price_changes.append(None)
        else:
            change = get_price_change(ticker, date_str)
            price_changes.append(change)
            print(f"{date_str} | {ticker} | {change}%")

    df["price_change_percent"] = price_changes
    df.to_csv(output_file, index=False)
    print("\nOutput saved to:", output_file)

if __name__ == "__main__":
    add_price_change_to_csv("data/articles_with_finbert_sentiment.csv", "data/articles_with_price_change.csv")
