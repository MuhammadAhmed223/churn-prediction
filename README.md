# Churn Prediction

This is a machine learning project that predicts customer churn for a telecom company using a trained classification model. It includes a GUI-based interface built with Tkinter, allowing users to input customer details and receive churn predictions.

## 📁 Project Structure

churn-prediction/
├── app/
│ ├── app.py # CLI or base script
│ └── gui_app.py # Tkinter GUI application
├── models/
│ ├── best_model.pkl # Trained machine learning model
│ ├── scaler.pkl # StandardScaler object used for preprocessing
│ └── feature_names.pkl # Ordered list of training feature names
├── notebooks/
│ └── 01_training.ipynb # Jupyter notebook for data cleaning, EDA, model training & evaluation
├── data/
│ └── Telco-Customer-Churn.csv # Raw dataset
├── requirements.txt # Python package dependencies
├── README.md # This file
└── .gitignore # Files and folders to ignore in version control



---

## 🧠 Model Summary

The model is trained using customer information such as:

- Demographics: Gender, SeniorCitizen, Partner, Dependents
- Account Information: Tenure, Contract type, Monthly Charges, etc.
- Services Signed Up: Phone, Internet, Streaming, etc.
- Payment Info: Payment method, paperless billing, etc.

The model uses standard preprocessing (`StandardScaler`) and a trained classifier (e.g., RandomForest, Logistic Regression, etc.) saved as `best_model.pkl`.

---

## 💻 GUI Application (Tkinter)

A Tkinter-based user-friendly GUI (`gui_app.py`) lets users input features and predict churn:

### 🧾 Features Included:

- Gender
- Senior Citizen (0 or 1)
- Partner (Yes or No)
- Dependents (Yes or No)
- Tenure (numeric)
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

### ✅ Example Input (for testing):

## ```plaintext
Gender: Female
Senior Citizen: 0
Partner: Yes
Dependents: No
Tenure: 12
PhoneService: Yes
MultipleLines: No
InternetService: Fiber optic
OnlineSecurity: No
OnlineBackup: No
DeviceProtection: No
TechSupport: No
StreamingTV: Yes
StreamingMovies: No
Contract: Month-to-month
PaperlessBilling: Yes
PaymentMethod: Electronic check
MonthlyCharges: 75.3
TotalCharges: 864.5

## 📦 Installation

git clone https://github.com/MuhammadAhmed223/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt

## 🚀 Running the GUI

cd app
python gui_app.py

