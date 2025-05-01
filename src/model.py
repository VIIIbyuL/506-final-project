"""
    Model evaluation and visualization for the alignment prediction task.
    This script includes functions for loading data, preparing features,
    and evaluating the model using different methods: random split,
    time-based split, and k-fold cross-validation.
    It also includes functions for visualizing the results, such as
    confusion matrices and feature importances.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error

# visualizations
def save_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Aligned", "Aligned"])
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"visual/{filename}")
    plt.close()

def save_feature_importance(clf, feature_names, filename):
    importances = clf.feature_importances_
    plt.barh(feature_names, importances)
    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"visual/{filename}")
    plt.close()

def save_kfold_scores(scores, filename="kfold_scores.png"):
    plt.bar(range(1, len(scores) + 1), scores)
    plt.ylim(0, 1)
    plt.title("5-Fold Cross-Validation Accuracy")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(f"visual/{filename}")
    plt.close()

def plot_sentiment_counts_per_company(df, company_name, filename):
    company_df = df[df["Company"] == company_name]
    sentiment_counts = company_df["finbert_sentiment_label"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['orange', 'red', 'green'])
    plt.title(f'Sentiment Counts for {company_name}')
    plt.xlabel('Sentiment Labels')
    plt.ylabel('Counts')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"visual/{filename}")
    plt.close()

def plot_sentiment_trends_over_time(df, sentiment_column='finbert_sentiment_label'):
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    sentiment_counts = df.groupby(['Date', 'Company', sentiment_column]).size().unstack(fill_value=0)
    companies = sentiment_counts.index.get_level_values('Company').unique()
    num_companies = len(companies)
    fig, axes = plt.subplots(num_companies, 1, figsize=(10, 4 * num_companies), sharex=True)

    # Determine the maximum Y-axis limit across all companies
    max_count = sentiment_counts.max().max()

    # Plot data for each company
    for ax, company in zip(axes, companies):
        company_data = sentiment_counts.xs(company, level='Company')
        company_data.plot(kind='line', marker='o', ax=ax)
        ax.set_title(f'Sentiment Trends Over Time for {company}')
        ax.set_ylabel('Number of Sentiments')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_ylim(0, max_count)
        ax.legend(title='Sentiment Labels', loc='upper left', labels=company_data.columns)
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("visual/sentiment_trends_over_time.png")
    plt.close()

def plot_price_change_counts_for_company(df, company_name, filename):
    # Filter the DataFrame for the specified company
    company_df = df[df['Company'] == company_name]

    # Count occurrences of each price_change_percent for the company
    price_change_counts = company_df['price_change_percent'].value_counts()

    # Sort the price change counts by the index (price_change_percent) in ascending order
    price_change_counts = price_change_counts.sort_index()

    # Create a bar plot
    price_change_counts.plot(kind='bar', figsize=(10, 6), width=0.8, color='skyblue')

    plt.title(f'Count of Price Change Percent for {company_name}')
    plt.xlabel('Price Change Percent')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"visual/{filename}")
    plt.close()

# load + prep

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df["sentiment_encoded"] = df["finbert_sentiment_label"].map(sentiment_map)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # Drop rows with any missing critical features
    df = df.dropna(subset=[
        "alignment_label", "sentiment_encoded", "finbert_confidence_percent",
        "prev_day_change", "sp500_change", "nasdaq_change", "vix_change"
    ])

    return df


def prepare_features(df):
    feature_cols = [
        "sentiment_encoded", "finbert_confidence_percent",
        "day_of_week", "month",
        "prev_day_change", "sp500_change", "nasdaq_change", "vix_change"
    ]
    X = df[feature_cols]
    y = df["alignment_label"]
    return X, y

# evals

def random_split_evaluation(X, y):
    print("\n [1] RANDOM TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    save_confusion(y_test, y_pred, "Random Split: Confusion Matrix", "confusion_random.png")
    save_feature_importance(clf, X.columns, "feature_importance_random.png")

def time_split_evaluation(df):
    print("\n [2] TIME-BASED SPLIT")
    df_sorted = df.sort_values("Date")
    split_index = int(len(df_sorted) * 0.8)
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]
    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    save_confusion(y_test, y_pred, "Time-Based Split: Confusion Matrix", "confusion_time.png")
    save_feature_importance(clf, X_train.columns, "feature_importance_time.png")

def kfold_evaluation(X, y):
    print("\n [3] K-FOLD CROSS-VALIDATION")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    scores = cross_val_score(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        ),
        X, y, cv=5, scoring='accuracy'
    )

    print("K-Fold Scores:", scores)
    print("Mean Accuracy:", scores.mean())
    save_kfold_scores(scores)

# preparing the features 

def prepare_features_for_prediction(df, lookback_window=3):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    
    # Ensure numeric columns are properly typed
    df["finbert_confidence_percent"] = pd.to_numeric(df["finbert_confidence_percent"], errors="coerce")
    df["price_change_percent"] = pd.to_numeric(df["price_change_percent"], errors="coerce")
    df["prev_day_change"] = pd.to_numeric(df["prev_day_change"], errors="coerce")
    df["sp500_change"] = pd.to_numeric(df["sp500_change"], errors="coerce")
    df["nasdaq_change"] = pd.to_numeric(df["nasdaq_change"], errors="coerce")
    df["vix_change"] = pd.to_numeric(df["vix_change"], errors="coerce")

    features = []
    targets = []
    dates = sorted(df["Date"].unique())
    last_feature_row = None

    for i in range(lookback_window, len(dates) - 1):  # Training pairs
        current_date = dates[i]
        next_date = dates[i + 1]
        window_dates = dates[i - lookback_window:i]

        window_df = df[df["Date"].isin(window_dates)]
        current_df = df[df["Date"] == current_date]
        next_df = df[df["Date"] == next_date]

        if len(current_df) == 0 or len(next_df) == 0 or len(window_df) == 0:
            continue

        row = compute_feature_row(current_df, window_df)
        features.append(row)
        targets.append(next_df["price_change_percent"].iloc[0])

    # Last row for predicting the *next* day's change
    if len(dates) >= lookback_window:
        current_date = dates[-1]
        window_dates = dates[-lookback_window:]
        window_df = df[df["Date"].isin(window_dates)]
        current_df = df[df["Date"] == current_date]

        if not current_df.empty and not window_df.empty:
            last_feature_row = compute_feature_row(current_df, window_df)

    return np.array(features), np.array(targets), last_feature_row

def compute_feature_row(current_df, window_df):
    confidence = current_df["finbert_confidence_percent"].mean()
    sp500 = current_df["sp500_change"].mean()
    nasdaq = current_df["nasdaq_change"].mean()
    vix = current_df["vix_change"].mean()

    positive_ratio = (current_df["finbert_sentiment_label"] == "positive").mean()
    negative_ratio = (current_df["finbert_sentiment_label"] == "negative").mean()
    neutral_ratio = (current_df["finbert_sentiment_label"] == "neutral").mean()

    article_volume_today = len(current_df)

    hist_price_mean = window_df["price_change_percent"].mean()
    hist_price_std = window_df["price_change_percent"].std()
    hist_conf_mean = window_df["finbert_confidence_percent"].mean()
    hist_volume_mean = window_df.groupby("Date").size().mean()

    volume_delta = article_volume_today - hist_volume_mean

    return [
        confidence,
        sp500,
        nasdaq,
        vix,
        positive_ratio,
        negative_ratio,
        neutral_ratio,
        hist_price_mean,
        hist_price_std,
        hist_conf_mean,
        article_volume_today,
        volume_delta,
    ]


def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Prediction for each company

def predict_for_each_company(df):
    print("\n [4] NEXT DAY OPENER PREDICTIONS")
    companies = df["Company"].unique()
    
    predictions = []
    
    for company in companies:
        company_df = df[df["Company"] == company]
        company_df = company_df.sort_values("Date")
        X, y, last_feature_row = prepare_features_for_prediction(company_df, 3)

        if len(X) > 0:  # Ensure that we have at least one valid training example
            # Train model for the company
            model = train_model(X, y)
            
            # Get predicted trend (Up or Down) for the *next* day after the last available data
            if last_feature_row is not None:
                last_feature_row = np.array(last_feature_row)
                predicted_price_change = model.predict(last_feature_row.reshape(1, -1))[0]
                trend = "Up" if predicted_price_change > 0 else "Down"
                
                # Store predictions
                predictions.append({
                    "Company": company,
                    "Predicted Trend": trend,
                    "Predicted Price Change (%)": predicted_price_change
                })
            else:
                print(f"No valid data to predict for {company}")
        else:
            print(f"No training data for {company}")
    
    return predictions

# main

def main():
    df = load_data("data/articles_with_alignment_labels_v2.csv")
    X, y = prepare_features(df)

    random_split_evaluation(X, y)
    time_split_evaluation(df)
    kfold_evaluation(X, y)
    # predict_next_day(df, num_days=7)
    plot_sentiment_counts_per_company(df, "Apple", filename="sentiment_counts_apple.png")
    plot_sentiment_counts_per_company(df, "Amazon", filename="sentiment_counts_amazon.png")
    plot_sentiment_counts_per_company(df, "Tesla", filename="sentiment_counts_tesla.png")
    plot_sentiment_trends_over_time(df)
    plot_price_change_counts_for_company(df, "Apple", filename="price_change_counts_apple.png")
    plot_price_change_counts_for_company(df, "Amazon", filename="price_change_counts_amazon.png")
    plot_price_change_counts_for_company(df, "Tesla", filename="price_change_counts_tesla.png")   
    
    # Predict for each company
    predictions = predict_for_each_company(df)
    
    # Print the predictions for each company
    for prediction in predictions:
        print(f"Company: {prediction['Company']}\nPredicted Trend: {prediction['Predicted Trend']}\nPredicted Price Change: {prediction['Predicted Price Change (%)']}%")

if __name__ == "__main__":
    main()
