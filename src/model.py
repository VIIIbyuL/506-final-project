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
    df = df.dropna(subset=["alignment_label", "finbert_sentiment_label", "finbert_confidence_percent", "Date"])
    df["Date"] = pd.to_datetime(df["Date"])
    sentiment_map = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df["sentiment_encoded"] = df["finbert_sentiment_label"].map(sentiment_map)
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    return df

def prepare_features(df):
    feature_cols = ["sentiment_encoded", "finbert_confidence_percent", "day_of_week", "month"]
    X = df[feature_cols]
    y = df["alignment_label"]
    return X, y

# evals

def random_split_evaluation(X, y):
    print("\n [1] RANDOM TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(random_state=42)
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

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    save_confusion(y_test, y_pred, "Time-Based Split: Confusion Matrix", "confusion_time.png")
    save_feature_importance(clf, X_train.columns, "feature_importance_time.png")

def kfold_evaluation(X, y):
    print("\n [3] K-FOLD CROSS-VALIDATION")
    clf = RandomForestClassifier(random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("K-Fold Scores:", scores)
    print("Mean Accuracy:", scores.mean())
    save_kfold_scores(scores)

def prepare_features_for_prediction(df, days_back):
    features = []
    target = []
    
    # Sort dataframe by Date in ascending order (ensure correct order)
    df = df.sort_values("Date")
    
    # Get the latest `days_back` distinct dates
    latest_dates = df['Date'].drop_duplicates().tail(days_back).values
    
    # Loop through the latest `days_back` dates
    for date in latest_dates:
        # Filter data for the current date
        day_window = df[df['Date'] == date]
        # Ensure there are articles for the current day
        if len(day_window) > 0:
            # Average the features across all articles for the day
            confidence_values = day_window['finbert_confidence_percent'].mean()
            day_of_week_values = day_window['day_of_week'].mode()[0]  # Take the most frequent day of week
            month_values = day_window['month'].mode()[0]  # Take the most frequent month
            
            # Count the number of positive, neutral, and negative articles
            positive_count = day_window[day_window['sentiment_encoded'] == 1].shape[0]
            negative_count = day_window[day_window['sentiment_encoded'] == -1].shape[0]
            neutral_count = day_window[day_window['sentiment_encoded'] == 0].shape[0]

            # Calculate the total number of articles for normalization
            total_articles = len(day_window)

            # Compute the proportions of each sentiment
            positive_ratio = positive_count / total_articles
            negative_ratio = negative_count / total_articles
            neutral_ratio = neutral_count / total_articles

            # Now include these proportions as features
            features.append(np.array([positive_ratio, negative_ratio, neutral_ratio, confidence_values, day_of_week_values, month_values]))
            target.append(day_window['price_change_percent'].iloc[0])  # Use the first price_change_percent for this date (assuming it's the same for the day)
    
    features = np.array(features)
    target = np.array(target)
    return features, target


def train_model(X_train, y_train):
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model

# Prediction for each company

def predict_for_each_company(df, days_back):
    print("\n [4] NEXT DAY PREDICTIONS")
    companies = df["Company"].unique()
    
    predictions = []
    
    for company in companies:
        company_df = df[df["Company"] == company]
        X, y = prepare_features_for_prediction(company_df, days_back)
        
        # Train model for the company
        model = train_model(X, y)
        
        # Get predicted trend (Up or Down)
        predicted_price_change = model.predict(X[-1].reshape(1, -1))[0]
        trend = "Up" if predicted_price_change > 0 else "Down"
        
        # Store predictions
        predictions.append({
            "Company": company,
            "Predicted Trend": trend,
            "Predicted Price Change (%)": predicted_price_change
        })
    
    return predictions

# main

def main():
    df = load_data("data/articles_with_alignment_labels.csv")
    X, y = prepare_features(df)

    random_split_evaluation(X, y)
    time_split_evaluation(df)
    kfold_evaluation(X, y)
    predict_next_day(df, num_days=7)
    plot_sentiment_counts_per_company(df, "Apple", filename="sentiment_counts_apple.png")
    plot_sentiment_counts_per_company(df, "Amazon", filename="sentiment_counts_amazon.png")
    plot_sentiment_counts_per_company(df, "Tesla", filename="sentiment_counts_tesla.png")
    plot_sentiment_trends_over_time(df)
    plot_price_change_counts_for_company(df, "Apple", filename="price_change_counts_apple.png")
    plot_price_change_counts_for_company(df, "Amazon", filename="price_change_counts_amazon.png")
    plot_price_change_counts_for_company(df, "Tesla", filename="price_change_counts_tesla.png")   
    
    # Predict for each company
    predictions = predict_for_each_company(df, 7)
    
    # Print the predictions for each company
    for prediction in predictions:
        print(f"Company: {prediction['Company']}\nPredicted Trend: {prediction['Predicted Trend']}\nPredicted Price Change: {prediction['Predicted Price Change (%)']}%")

if __name__ == "__main__":
    main()
