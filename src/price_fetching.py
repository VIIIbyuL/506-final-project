"""
This script fetches stock price change percent and adds market context features
(S&P 500, NASDAQ, VIX, and previous day price change) to a CSV of articles.
"""

import pandas as pd
import yfinance as yf
import datetime

def get_price_change(ticker, date_str):
    dt = pd.to_datetime(date_str)

    # Fetch up to 5 days of data to ensure at least 2 valid trading days
    data = yf.download(ticker, start=dt.strftime('%Y-%m-%d'),
                       end=(dt + datetime.timedelta(days=5)).strftime('%Y-%m-%d'),
                       progress=False)
    data = data[data.index >= dt]

    if data.empty or len(data) < 2:
        return None, None

    # Get change from first trading day open to second day close
    open_price = data.iloc[0]['Open']
    close_price = data.iloc[1]['Close']
    prev_open = data.iloc[0]['Open']
    prev_close = data.iloc[0]['Close']

    price_change = ((close_price - open_price) / open_price) * 100
    prev_day_change = ((prev_close - prev_open) / prev_open) * 100
    return price_change, prev_day_change

def get_index_change(ticker, date):
    data = yf.download(ticker, start=date.strftime('%Y-%m-%d'),
                       end=(date + datetime.timedelta(days=5)).strftime('%Y-%m-%d'),
                       progress=False)
    data = data[data.index >= date]

    if data.empty or len(data) < 2:
        return None

    open_val = data.iloc[0]['Open']
    close_val = data.iloc[1]['Close']
    return ((close_val - open_val) / open_val) * 100

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
    prev_day_changes = []
    sp500_changes = []
    nasdaq_changes = []
    vix_changes = []

    for idx, row in df.iterrows():
        ticker = row["Ticker"]
        date_str = row["Date"]

        if pd.isna(ticker) or pd.isna(date_str):
            price_changes.append(None)
            prev_day_changes.append(None)
            sp500_changes.append(None)
            nasdaq_changes.append(None)
            vix_changes.append(None)
            continue

        try:
            dt = pd.to_datetime(date_str)
        except Exception as e:
            print(f"Skipping row {idx} due to date parsing error: {e}")
            price_changes.append(None)
            prev_day_changes.append(None)
            sp500_changes.append(None)
            nasdaq_changes.append(None)
            vix_changes.append(None)
            continue

        # Get values
        change, prev = get_price_change(ticker, date_str)
        sp500 = get_index_change("^GSPC", dt)
        nasdaq = get_index_change("^IXIC", dt)
        vix = get_index_change("^VIX", dt)

        # Ensure plain floats (fixes the string Series problem)
        change = float(change) if change is not None else None
        prev = float(prev) if prev is not None else None
        sp500 = float(sp500) if sp500 is not None else None
        nasdaq = float(nasdaq) if nasdaq is not None else None
        vix = float(vix) if vix is not None else None

        price_changes.append(change)
        prev_day_changes.append(prev)
        sp500_changes.append(sp500)
        nasdaq_changes.append(nasdaq)
        vix_changes.append(vix)

        print(f"{date_str} | {ticker} | Δ: {change}% | Prev: {prev}% | S&P500: {sp500}% | NASDAQ: {nasdaq}% | VIX: {vix}%")

    df["price_change_percent"] = price_changes
    df["prev_day_change"] = prev_day_changes
    df["sp500_change"] = sp500_changes
    df["nasdaq_change"] = nasdaq_changes
    df["vix_change"] = vix_changes

    df.to_csv(output_file, index=False)
    print(f"\nOutput saved to: {output_file}")

if __name__ == "__main__":
    add_price_change_to_csv(
        "../data/articles_with_finbert_sentiment_v2.csv",
        "../data/articles_with_price_change_v2.csv"
    )
