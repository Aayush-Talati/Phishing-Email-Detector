Phishing Email Detector

A machine learning web app that detects whether an email is likely to be phishing or legitimate.

This project uses Python, scikit-learn, and Streamlit to train a phishing email classifier on multiple datasets. Users can paste an email into the web app and receive a phishing risk score, prediction, explanation, and safety recommendation.

Project Overview

Phishing emails are fake or misleading messages designed to trick users into clicking malicious links, downloading dangerous attachments, or giving away sensitive information such as passwords, bank details, or login credentials.

This project combines artificial intelligence and cybersecurity by using machine learning to identify suspicious email patterns.

The app analyzes email text, links, suspicious keywords, urgency language, and other phishing indicators to predict whether a message is safe or suspicious.

Features
Detects phishing and legitimate emails
Trains on multiple email datasets
Uses TF-IDF natural language processing
Extracts cybersecurity-specific features
Predicts phishing probability
Shows a risk score from 0 to 100
Provides Low, Medium, High, or Very High Risk labels
Explains why an email may be suspicious
Displays detected URLs
Gives safety recommendations
Includes a Streamlit web interface
Technologies Used
Python
pandas
NumPy
scikit-learn
Streamlit
joblib
Project Structure

Phishing-Email-Detector/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│ ├── phishing_email.csv
│ ├── CEAS_08.csv
│ ├── Enron.csv
│ ├── Ling.csv
│ ├── Nazario.csv
│ ├── Nigerian_Fraud.csv
│ └── SpamAssasin.csv
└── models/
└── phishguard_model.joblib

File Descriptions
app.py

This file runs the Streamlit web app.

It loads the trained machine learning model, allows users to paste an email, predicts whether the email is phishing, and displays the result.

The app shows:

phishing probability
risk score
prediction result
suspicious indicators
detected URLs
safety recommendation
train_model.py

This file trains the phishing detection model.

It performs these steps:

Loads CSV files from the data folder
Combines email subject, body, sender, and URL fields when available
Cleans and normalizes the labels
Converts email text into TF-IDF features
Extracts cybersecurity-based features
Splits the data into training and testing sets
Trains the machine learning classifier
Evaluates the model
Saves the trained model to the models folder
requirements.txt

This file contains the Python packages needed to run the project.

data/

This folder is used for the training datasets.

The dataset CSV files are not included in this GitHub repository because they may be large and may have separate licensing requirements.

models/

This folder stores the trained machine learning model.

The main model file is:

models/phishguard_model.joblib

This file is created after training the model.

How the Model Works

The project uses two main types of features:

TF-IDF text features
Cybersecurity-based features
TF-IDF Text Features

TF-IDF stands for Term Frequency-Inverse Document Frequency.

It converts email text into numerical values so the machine learning model can understand which words and phrases are important.

For example, phishing emails often contain words or phrases like:

urgent
verify your account
password
click here
account suspended
payment failed

TF-IDF helps the model learn which words and phrases are more common in phishing emails compared to legitimate emails.

Cybersecurity-Based Features

The model also extracts custom cybersecurity features from each email.

These include:

word count
character count
number of exclamation marks
number of question marks
uppercase letter ratio
number of URLs
suspicious keyword count
whether a URL uses an IP address
whether a URL uses a link shortener
whether the email contains money-related terms
whether the email contains credential-related terms

These features help the model detect common phishing techniques such as urgency, fake login requests, suspicious links, and financial scams.

Machine Learning Model

The project uses an SGDClassifier with logistic loss.

This works similarly to logistic regression and allows the model to output a phishing probability.

The classifier predicts:

0 = legitimate email
1 = phishing email

The phishing probability is converted into a risk score:

Risk Score = phishing probability x 100

Example:

Phishing probability: 0.87
Risk score: 87/100
Risk label: Very High Risk

Risk Levels

The app uses these risk levels:

0-34 = Low Risk
35-59 = Medium Risk
60-79 = High Risk
80-100 = Very High Risk

Datasets

This project expects the following datasets inside the data folder:

phishing_email.csv
CEAS_08.csv
Enron.csv
Ling.csv
Nazario.csv
Nigerian_Fraud.csv
SpamAssasin.csv

Some datasets use a column called text_combined.

Other datasets use columns such as:

subject
body
sender
urls

The training script automatically combines the available columns into one email text field.

Dataset Access

The datasets are not included in this repository because they may be large and may have separate licensing requirements.

To train the model, download the datasets separately and place them inside the data folder.

Expected file locations:

data/phishing_email.csv
data/CEAS_08.csv
data/Enron.csv
data/Ling.csv
data/Nazario.csv
data/Nigerian_Fraud.csv
data/SpamAssasin.csv

After adding the datasets, run the training command to create the model file.

Downloading the Project

To download this project from GitHub, run:

git clone https://github.com/Aayush-Talati/Phishing-Email-Detector.git

Then move into the project folder:

cd Phishing-Email-Detector

Installing Requirements

Install the required Python packages:

pip install -r requirements.txt

On Windows, if that does not work, use:

python -m pip install -r requirements.txt

Training the Model

Make sure the datasets are inside the data folder.

To train the model using all rows, run:

python train_model.py --data-dir data --model-path models/phishguard_model.joblib

For a faster test run, use:

python train_model.py --data-dir data --model-path models/phishguard_model.joblib --max-rows-per-file 10000

After training, the model will be saved here:

models/phishguard_model.joblib

Running the App Locally

Run the Streamlit app with:

python -m streamlit run app.py

Streamlit will show a local URL in the terminal, usually:

http://localhost:8501

Open that link in your browser.

How to Use the App
Paste an email into the text box.
Adjust the phishing threshold if needed.
Click Analyze Email.
Review the phishing probability.
Check the risk label.
Read the explanation.
Look at the detected URLs.
Follow the safety recommendation.
Example Phishing Email

URGENT: Your account has been suspended!

We detected unusual activity on your bank account. Click here immediately to verify your password:
http://secure-login.verify-account.example.com/login

Failure to act now will result in permanent account deletion.

Expected result:

High Risk or Very High Risk

Example Legitimate Email

Hi,

Can we move our project meeting to Thursday at 2 PM? I finished the notes and will share them before class.

Thanks.

Expected result:

Low Risk

Deploying to Streamlit Community Cloud

To deploy the app publicly:

Push the project to GitHub.
Go to Streamlit Community Cloud.
Create a new app.
Connect your GitHub account.
Select this repository:

Aayush-Talati/Phishing-Email-Detector

Select the branch:

main

Set the main file path to:

app.py

Click Deploy.
Important Deployment Note

Streamlit Community Cloud needs access to the trained model file.

Common Issues
Streamlit command is not recognized

Use this command instead:

python -m streamlit run app.py

Model file is missing

Train the model first:

python train_model.py --data-dir data --model-path models/phishguard_model.joblib --max-rows-per-file 10000

Then check that the model exists:

dir models

You should see:

phishguard_model.joblib

Git says the model file does not match any files

That means the model file does not exist yet.

Train the model first, then add it to Git:

git add models/phishguard_model.joblib

Streamlit Cloud cannot find the app

Make sure the main file path is:

app.py

Streamlit Cloud cannot find the model

Make sure this file is committed and pushed to GitHub:

models/phishguard_model.joblib

Limitations

This project is for educational and research purposes only.

Limitations include:

The model depends on the quality of the datasets
It may not detect every phishing email
It may incorrectly flag some legitimate emails
It does not check live domain reputation
It does not scan attachments
It does not verify sender authentication records like SPF, DKIM, or DMARC
Future Improvements

Possible future improvements include:

email header analysis
sender spoofing detection
domain reputation checking
attachment scanning
SHAP or LIME model explanations
evaluation charts
browser extension support
user feedback for improving predictions
Disclaimer

This project is for educational and research purposes only. It should not be used as the only security system for detecting phishing emails. Always verify suspicious emails through official sources before clicking links, downloading attachments, or entering credentials.
