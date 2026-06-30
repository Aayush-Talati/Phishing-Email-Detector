# Phishing Email Detector

A machine learning project that detects phishing emails using natural language processing and cybersecurity-based feature extraction.

The project trains on multiple phishing and legitimate email datasets, builds a machine learning classifier, and provides a Streamlit web app where users can paste an email and receive a phishing risk score with explanations.

## Features

- Detects phishing and legitimate emails
- Trains on multiple CSV datasets
- Uses TF-IDF text vectorization
- Extracts cybersecurity-specific features, including:
  - URL count
  - suspicious keywords
  - credential-related terms
  - money-related terms
  - URL shorteners
  - IP-based URLs
  - excessive punctuation
  - uppercase ratio
- Displays phishing probability
- Provides a risk level
- Explains why an email may be suspicious
- Includes a Streamlit web interface

## Project Structure

```text
Phishing Emails/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── phishing_email.csv
│   ├── CEAS_08.csv
│   ├── Enron.csv
│   ├── Ling.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   └── SpamAssasin.csv
└── models/
    └── phishguard_model.joblib