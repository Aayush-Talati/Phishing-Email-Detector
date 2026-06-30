import os
import re
import joblib
import numpy as np
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin


# Path to the trained model created by train_model.py
MODEL_PATH = "models/phishguard_model.joblib"


# Regular expression used to detect links in pasted emails.
URL_PATTERN = r"https?://[^\s]+|www\.[^\s]+"


# These words are commonly seen in phishing emails.
# The app uses them to explain suspicious results.
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
    This class selects the email text column.

    It must also exist in app.py because the saved model pipeline uses this class.
    Without it, joblib may not be able to load the model.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X["text"].fillna("").astype(str)


class CyberFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts cybersecurity-related features from the email.

    This must match the CyberFeatureExtractor from train_model.py.
    If the features are different here, the saved model may not work correctly.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        texts = X["text"].fillna("").astype(str).tolist()
        feature_rows = []

        for text in texts:
            lower = text.lower()
            urls = re.findall(URL_PATTERN, text)

            # Basic email statistics
            word_count = len(text.split())
            char_count = len(text)
            exclamation_count = text.count("!")
            question_count = text.count("?")
            uppercase_count = sum(1 for c in text if c.isupper())

            # Ratio of uppercase letters to total characters
            uppercase_ratio = uppercase_count / max(char_count, 1)

            # Number of links in the email
            url_count = len(urls)

            # Count suspicious phishing-related words
            suspicious_keyword_count = sum(
                1 for keyword in SUSPICIOUS_KEYWORDS if keyword in lower
            )

            # Checks if a URL uses an IP address
            has_ip_url = int(
                any(re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", url) for url in urls)
            )

            # Checks if a URL uses a link shortener
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

            # Checks for banking, payment, and money-related language
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

            # Checks for login or identity verification language
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

            # These features must stay in the same order as train_model.py
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


def extract_urls(text):
    """
    Finds and returns all URLs in the email text.
    """
    return re.findall(URL_PATTERN, text)


def get_explanations(text):
    """
    Creates human-readable explanations for why an email may look suspicious.

    The machine learning model gives the probability, but this function helps
    the user understand the warning signs.
    """
    explanations = []
    lower = text.lower()
    urls = extract_urls(text)

    # Find suspicious words that appear in the email
    found_keywords = [
        keyword for keyword in SUSPICIOUS_KEYWORDS if keyword in lower
    ]

    if found_keywords:
        explanations.append(
            "Suspicious phishing-related words found: "
            + ", ".join(found_keywords[:8])
        )

    if urls:
        explanations.append(f"The message contains {len(urls)} link(s).")

    # Analyze each detected URL
    for url in urls:
        url_lower = url.lower()

        # IP-based links are suspicious because real companies usually use domains.
        if re.search(r"https?://\d{1,3}(\.\d{1,3}){3}", url):
            explanations.append(
                f"Suspicious URL uses an IP address instead of a normal domain: {url}"
            )

        # Shortened links can hide the real destination.
        if any(shortener in url_lower for shortener in ["bit.ly", "tinyurl.com", "t.co", "goo.gl"]):
            explanations.append(
                f"URL shortener detected, which can hide the real destination: {url}"
            )

        # Login/account words in URLs are common in phishing.
        if any(term in url_lower for term in ["login", "verify", "secure", "account", "update"]):
            explanations.append(
                f"URL contains account/security-related wording: {url}"
            )

        # Many subdomains can be used to make a fake domain look official.
        clean_domain = (
            url_lower.replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .split("/")[0]
        )

        if clean_domain.count(".") >= 3:
            explanations.append(
                f"URL has many subdomains, which can be suspicious: {url}"
            )

    if text.count("!") >= 3:
        explanations.append("The message uses many exclamation marks.")

    if len(text.split()) < 10:
        explanations.append("The message is very short and lacks context.")

    if any(term in lower for term in ["password", "login", "verify", "confirm"]):
        explanations.append("The message asks for login or identity verification.")

    if any(term in lower for term in ["urgent", "immediately", "act now", "limited time"]):
        explanations.append("The message uses urgency or pressure tactics.")

    if not explanations:
        explanations.append("No obvious phishing indicators were found.")

    return explanations


def risk_label(score):
    """
    Converts a numeric risk score into a readable risk label.
    """
    if score >= 80:
        return "Very High Risk"

    if score >= 60:
        return "High Risk"

    if score >= 35:
        return "Medium Risk"

    return "Low Risk"


def recommendation(score):
    """
    Returns safety advice based on the phishing risk score.
    """
    if score >= 60:
        return (
            "Do not click links, download attachments, or enter credentials. "
            "Verify the message using an official website or trusted contact."
        )

    if score >= 35:
        return (
            "Be careful. Check the sender, links, and request before taking action."
        )

    return (
        "This appears lower risk, but still verify links and attachments before trusting it."
    )


@st.cache_resource
def load_model():
    """
    Loads the saved machine learning model.

    Streamlit caches this so the model does not reload every time the page updates.
    """
    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


# Page settings for the Streamlit app
st.set_page_config(
    page_title="PhishGuard AI",
    page_icon="🛡️",
    layout="centered",
)


# App title and description
st.title("PhishGuard AI")
st.subheader("Intermediate phishing email detector")

st.write(
    "Paste an email below. The model will estimate phishing risk and explain warning signs."
)


# Load the trained model
model = load_model()


# Stop the app if the trained model is missing
if model is None:
    st.error(
        "Model not found. Train it first by running:\n\n"
        "`python train_model.py --data-dir data --model-path models/phishguard_model.joblib`"
    )
    st.stop()


# Default example shown in the text box
example_email = """URGENT: Your account has been suspended!

We detected unusual activity on your account. Click here immediately to verify your password:
http://secure-login.verify-account.example.com/login

Failure to act now will result in permanent account deletion.
"""


# Text box where users paste an email
email_text = st.text_area(
    "Email text",
    value=example_email,
    height=260,
)


# The user can adjust how strict the model should be.
# Lower threshold = more emails marked suspicious.
# Higher threshold = fewer emails marked suspicious.
threshold = st.slider(
    "Phishing threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
)


# Analyze button
if st.button("Analyze Email"):
    if not email_text.strip():
        st.warning("Paste an email first.")
        st.stop()

    # The model expects a DataFrame with a column named text.
    import pandas as pd

    input_df = pd.DataFrame({"text": [email_text]})

    # Predict probability that the email is phishing.
    phishing_probability = model.predict_proba(input_df)[0][1]

    # Convert probability into a 0-100 score.
    score = int(phishing_probability * 100)

    # Compare the probability against the selected threshold.
    is_phishing = phishing_probability >= threshold

    # Convert score into a readable risk label.
    label = risk_label(score)

    st.markdown("---")
    st.header("Result")

    # Show result with color based on risk level.
    if score >= 60:
        st.error(f"{label}: {score}/100")
    elif score >= 35:
        st.warning(f"{label}: {score}/100")
    else:
        st.success(f"{label}: {score}/100")

    st.write(f"Phishing probability: **{phishing_probability:.2%}**")
    st.write(f"Threshold: **{threshold:.2f}**")

    if is_phishing:
        st.write("Prediction: **Phishing / Suspicious**")
    else:
        st.write("Prediction: **Safe / Legitimate**")

    # Explain suspicious indicators found in the message.
    st.subheader("Explanation")
    explanations = get_explanations(email_text)

    for explanation in explanations:
        st.write(f"- {explanation}")

    # Give user safety advice.
    st.subheader("Recommendation")
    st.write(recommendation(score))

    # Show all URLs found in the email.
    with st.expander("Detected URLs"):
        urls = extract_urls(email_text)

        if urls:
            for url in urls:
                st.write(url)
        else:
            st.write("No URLs found.")