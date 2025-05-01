import pandas as pd
from src.model import prepare_features

def test_prepare_features_on_sample_row():
    # Simulated single-row DataFrame
    data = {
        "Company": ["Apple"],
        "Date": ["2024-05-08"],
        "sentiment_encoded": [-1],
        "finbert_confidence_percent": [63.54],
        "day_of_week": [2],  # Wednesday
        "month": [5],        # May
        "prev_day_change": [-0.0601589317362548],
        "sp500_change": [0.8725144579136112],
        "nasdaq_change": [0.7407181825929241],
        "vix_change": [-4.154080062256782],
        "alignment_label": [0],
        "price_change_percent": [0.940659289762006],
        "price_direction": [1]  # You may add this during load_data
    }

    df = pd.DataFrame(data)
    X, y = prepare_features(df)


    assert not X.empty
    assert not y.empty
    assert list(X.columns) == [
        "sentiment_encoded", "finbert_confidence_percent",
        "day_of_week", "month",
        "prev_day_change", "sp500_change", "nasdaq_change", "vix_change"
    ]
    assert y.iloc[0] == 0
