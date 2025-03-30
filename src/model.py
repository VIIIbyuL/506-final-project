import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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

# main

def main():
    df = load_data("data/articles_with_alignment_labels.csv")
    X, y = prepare_features(df)

    random_split_evaluation(X, y)
    time_split_evaluation(df)
    kfold_evaluation(X, y)

if __name__ == "__main__":
    main()
