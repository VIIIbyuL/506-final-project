import pandas as pd

data = [
    [1, "Apple", "AAPL", "Apple reports record earnings in Q1 2024.", "2024-01-25 09:00"],
    [2, "Tesla", "TSLA", "Tesla faces probe over self-driving incidents.", "2024-02-10 14:00"],
    [3, "Amazon", "AMZN", "Amazon expands into new international markets.", "2024-03-05 11:30"],
    [4, "Microsoft", "MSFT", "Microsoft's Azure growth slows down this quarter.", "2024-03-15 16:45"],
    [5, "NVIDIA", "NVDA", "NVIDIA's AI chip revenue beats expectations.", "2024-03-20 08:00"],
]

df = pd.DataFrame(data, columns=["article_id", "company", "ticker", "text", "timestamp"])
df.to_csv("data/articles_scraped.csv", index=False)
