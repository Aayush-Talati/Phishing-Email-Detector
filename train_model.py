import argparse
import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler


# These are the dataset files the program will look for inside the data folder.
DATASET_FILES = [
    "phishing_email.csv",
    "CEAS_08.csv",
    "Enron.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv",
    "SpamAssasin.csv",
]


# Regular expression used to find URLs in email text.
URL_PATTERN = r"https?://[^\s]+|www\.[^\s]+"


# Common words and phrases often found in phishing emails.
# These are used for custom cybersecurity features.
SUSPICIOUS_KEYWORDS = [
    "urgent",
    "verify",
    "password",
    "suspended",
    "locked",
    "confirm",
    "billing",
    "credit card",
    "click here",
    "act now",
    "limited time",
    "winner",
    "prize",
    "claim",
    "payment failed",
    "login",
    "security alert",
    "unusual activity",
    "account deletion",
    "bank",
    "wire transfer",
    "invoice",
    "update your account",
]


class TextSelector(BaseEstimator, TransformerMixin):
    """
    This class selects the email text column from the dataset.

    It is used inside the scikit-learn pipeline so that the TF-IDF vectorizer
    only receives the text data.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X["text"].fillna("").astype(str)


class CyberFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    This class creates custom cybersecurity-related features.

    Instead of only using words from the email, this extracts extra signals like:
    - number of links
    - number of suspicious keywords
    - whether the email contains password/login terms
    - whether the URL uses an IP address
    - whether the URL uses a shortener
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X["text"].fillna("").astype(str).tolist()
        feature_rows = []

        for text in texts:
            lower = text.lower()
            urls = re.findall(URL_PATTERN, text)

            # Basic text statistics
            word_count = len(text.split())
            char_count = len(text)
            exclamation_count = text.count("!")
            question_count = text.count("?")
            uppercase_count = sum(1 for c in text if c.isupper())

            # Prevent division by zero if the message is empty
            uppercase_ratio = uppercase_count / max(char_count, 1)

            # URL-related features
            url_count = len(urls)

            # Count how many phishing-related words appear in the email
            suspicious_keyword_count = sum(
                1 for keyword in SUSPICIOUS_KEYWORDS if keyword in lower
            )

            # Checks if any URL uses an IP address instead of a normal domain name
            has_ip_url = int(
                any(re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", url) for url in urls)
            )

            # Checks if any URL uses a common URL shortener
            has_shortener = int(
                any(
                    shortener in url.lower()
                    for url in urls
                    for shortener in [
                        "bit.ly",
                        "tinyurl.com",
                        "t.co",
                        "goo.gl",
                        "ow.ly",
                        "is.gd",
                    ]
                )
            )

            # Checks if the email talks about money, payments, or banking
            has_money_terms = int(
                any(
                    term in lower
                    for term in [
                        "money",
                        "payment",
                        "bank",
                        "account",
                        "transfer",
                        "invoice",
                        "credit card",
                    ]
                )
            )

            # Checks if the email asks for login or identity verification
            has_credential_terms = int(
                any(
                    term in lower
                    for term in [
                        "password",
                        "login",
                        "verify",
                        "confirm",
                        "credentials",
                        "security code",
                    ]
                )
            )

            # Add all extracted features as one row
            feature_rows.append(
                [
                    word_count,
                    char_count,
                    exclamation_count,
                    question_count,
                    uppercase_ratio,
                    url_count,
                    suspicious_keyword_count,
                    has_ip_url,
                    has_shortener,
                    has_money_terms,
                    has_credential_terms,
                ]
            )

        return np.array(feature_rows, dtype=float)


def normalize_label(value):
    """
    Converts different label formats into a standard format.

    Final labels:
    1 = phishing, spam, malicious, fraud
    0 = safe, ham, legitimate, benign
    """
    if pd.isna(value):
        return np.nan

    value_str = str(value).strip().lower()

    phishing_values = {
        "1",
        "phishing",
        "spam",
        "malicious",
        "fraud",
        "bad",
        "true",
        "yes",
    }

    safe_values = {
        "0",
        "ham",
        "safe",
        "legitimate",
        "benign",
        "good",
        "false",
        "no",
    }

    if value_str in phishing_values:
        return 1

    if value_str in safe_values:
        return 0

    # Some datasets may store labels as 1.0 or 0.0
    try:
        numeric_value = int(float(value_str))

        if numeric_value == 1:
            return 1

        if numeric_value == 0:
            return 0

    except ValueError:
        pass

    # If the label is unknown, return NaN so it can be removed later
    return np.nan


def load_one_dataset(path, max_rows=None):
    """
    Loads one CSV dataset and converts it into a standard format.

    Output format:
    - text: combined email text
    - label: 0 or 1
    - source_file: dataset filename
    """
    print(f"Loading {path}")

    # nrows is optional. It lets you train faster while testing.
    df = pd.read_csv(path, nrows=max_rows)

    # Clean column names in case there are extra spaces
    df.columns = [col.strip() for col in df.columns]

    if "label" not in df.columns:
        raise ValueError(f"{path} does not have a label column.")

    # Some datasets already have one combined text column.
    if "text_combined" in df.columns:
        text = df["text_combined"].fillna("").astype(str)

    else:
        # Other datasets have separate subject/body/sender/url columns.
        parts = []

        if "subject" in df.columns:
            parts.append(df["subject"].fillna("").astype(str))

        if "body" in df.columns:
            parts.append(df["body"].fillna("").astype(str))

        if "sender" in df.columns:
            parts.append(" Sender: " + df["sender"].fillna("").astype(str))

        if "urls" in df.columns:
            parts.append(" URLs: " + df["urls"].fillna("").astype(str))

        if not parts:
            raise ValueError(f"{path} does not have usable text columns.")

        # Combine all available text-related columns into one text field.
        text = parts[0]

        for part in parts[1:]:
            text = text + " " + part

    # Normalize labels into 0 or 1
    labels = df["label"].apply(normalize_label)

    clean_df = pd.DataFrame(
        {
            "text": text,
            "label": labels,
            "source_file": os.path.basename(path),
        }
    )

    # Remove rows with invalid labels
    clean_df = clean_df.dropna(subset=["label"])
    clean_df["label"] = clean_df["label"].astype(int)

    # Remove rows with empty email text
    clean_df = clean_df[clean_df["text"].str.strip().str.len() > 0]

    return clean_df


def load_all_datasets(data_dir, max_rows_per_file=None):
    """
    Loads all datasets from the data folder and combines them into one DataFrame.
    """
    frames = []

    for filename in DATASET_FILES:
        path = os.path.join(data_dir, filename)

        # Skip files that are missing instead of crashing immediately.
        if not os.path.exists(path):
            print(f"Skipping missing file: {filename}")
            continue

        dataset = load_one_dataset(path, max_rows=max_rows_per_file)

        print("Label counts:")
        print(dataset["label"].value_counts())

        frames.append(dataset)

    if not frames:
        raise RuntimeError("No datasets were loaded. Check your data folder.")

    # Combine every dataset into one big dataset.
    combined = pd.concat(frames, ignore_index=True)

    # Shuffle the rows so emails from the same dataset are not grouped together.
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nCombined dataset:")
    print(combined["label"].value_counts())
    print(f"Total rows: {len(combined)}")

    return combined


def build_model():
    """
    Builds the full machine learning pipeline.

    The model uses two feature groups:
    1. TF-IDF text features from the email body
    2. Custom cybersecurity features from CyberFeatureExtractor
    """

    # Converts email text into numerical word/phrase features.
    text_pipeline = Pipeline(
        steps=[
            ("selector", TextSelector()),
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    max_features=60000,
                ),
            ),
        ]
    )

    # Extracts custom cybersecurity features and scales them.
    cyber_feature_pipeline = Pipeline(
        steps=[
            ("features", CyberFeatureExtractor()),
            ("scaler", StandardScaler()),
        ]
    )

    # Combines TF-IDF features and cybersecurity features.
    features = FeatureUnion(
        transformer_list=[
            ("text_features", text_pipeline),
            ("cyber_features", cyber_feature_pipeline),
        ]
    )

    # SGDClassifier with log_loss works like logistic regression
    # and supports probability predictions.
    model = Pipeline(
        steps=[
            ("features", features),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    penalty="l2",
                    alpha=0.0001,
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return model


def train(args):
    """
    Main training function.

    It loads data, splits into training/testing sets, trains the model,
    evaluates it, and saves it to a .joblib file.
    """

    df = load_all_datasets(
        data_dir=args.data_dir,
        max_rows_per_file=args.max_rows_per_file,
    )

    X = df[["text"]]
    y = df["label"]

    # Split the data into training and testing sets.
    # stratify=y keeps the phishing/safe ratio similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    model = build_model()

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("\nEvaluating model...")

    # Class predictions: 0 or 1
    predictions = model.predict(X_test)

    # Probability that each email is phishing
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=["Safe", "Phishing"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Create the models folder if it does not exist.
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Save the trained model so app.py can load it later.
    joblib.dump(model, args.model_path)

    print(f"\nModel saved to: {args.model_path}")


if __name__ == "__main__":
    # Command line arguments let you control paths and training options.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Folder containing your CSV datasets.",
    )

    parser.add_argument(
        "--model-path",
        default="models/phishguard_model.joblib",
        help="Where to save the trained model.",
    )

    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=None,
        help="Optional row limit per dataset for faster testing.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set size.",
    )

    args = parser.parse_args()
    train(args)