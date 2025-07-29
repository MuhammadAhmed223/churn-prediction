import joblib
import pandas as pd

# Load model, scaler, and feature names
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Sample input (make sure keys match original columns before encoding)
input_data = {
    'SeniorCitizen': 0,
    'tenure': 12,
    'MonthlyCharges': 70.35,
    'TotalCharges': 845.5,
    'Partner': 'Yes',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'PaperlessBilling': 'Yes',
    'InternetService': 'Fiber optic',
    'Contract': 'Month-to-month',
    'PaymentMethod': 'Electronic check',
    # Add all necessary categorical fields
}

# Convert to DataFrame
sample_df = pd.DataFrame([input_data])

# Encode binary and categorical fields
sample_df['Partner'] = sample_df['Partner'].map({'Yes': 1, 'No': 0})
sample_df['Dependents'] = sample_df['Dependents'].map({'Yes': 1, 'No': 0})
sample_df['PhoneService'] = sample_df['PhoneService'].map({'Yes': 1, 'No': 0})
sample_df['PaperlessBilling'] = sample_df['PaperlessBilling'].map({'Yes': 1, 'No': 0})

# One-hot encode remaining categorical fields
sample_df = pd.get_dummies(sample_df)

# Align with training features
sample_df = sample_df.reindex(columns=feature_names, fill_value=0)

# Scale input
scaled_input = scaler.transform(sample_df)

# Predict
prediction = model.predict(scaled_input)[0]
print("Churn Prediction:", "Yes" if prediction == 1 else "No")
