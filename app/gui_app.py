import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# Load model, scaler, and feature names
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

# Tkinter window
root = tk.Tk()
root.title("Customer Churn Prediction")
root.geometry("500x600")

fields = {}

# Function to simulate placeholders (hints)
def add_placeholder(entry, placeholder_text):
    entry.insert(0, placeholder_text)
    entry.config(fg='gray')

    def on_focus_in(event):
        if entry.get() == placeholder_text:
            entry.delete(0, tk.END)
            entry.config(fg='black')

    def on_focus_out(event):
        if entry.get() == '':
            entry.insert(0, placeholder_text)
            entry.config(fg='gray')

    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)

# Field creation with optional dropdowns
def create_input(label_text, row, values=None, placeholder=None):
    label = tk.Label(root, text=label_text)
    label.grid(row=row, column=0, padx=10, pady=5, sticky='w')

    if values:
        combo = ttk.Combobox(root, values=values, state='readonly')
        combo.grid(row=row, column=1, padx=10, pady=5)
        combo.current(0)
        fields[label_text] = combo
    else:
        entry = tk.Entry(root)
        entry.grid(row=row, column=1, padx=10, pady=5)
        fields[label_text] = entry
        if placeholder:
            add_placeholder(entry, placeholder)

# Binary fields
create_input("Partner", 0, ["Yes", "No"])
create_input("Dependents", 1, ["Yes", "No"])
create_input("PhoneService", 2, ["Yes", "No"])
create_input("PaperlessBilling", 3, ["Yes", "No"])

# Categorical fields
create_input("InternetService", 4, ["DSL", "Fiber optic", "No"])
create_input("Contract", 5, ["Month-to-month", "One year", "Two year"])
create_input("PaymentMethod", 6, [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
])

# Numeric fields with placeholders
create_input("SeniorCitizen", 7, placeholder="0 for No, 1 for Yes")
create_input("tenure", 8, placeholder="e.g. 12")
create_input("MonthlyCharges", 9, placeholder="e.g. 75.5")
create_input("TotalCharges", 10, placeholder="e.g. 1530.6")

# --- Predict Function ---
def predict():
    try:
        # Prepare input dict
        input_data = {
            'SeniorCitizen': int(fields['SeniorCitizen'].get()),
            'tenure': float(fields['tenure'].get()),
            'MonthlyCharges': float(fields['MonthlyCharges'].get()),
            'TotalCharges': float(fields['TotalCharges'].get()),
            'Partner': fields['Partner'].get(),
            'Dependents': fields['Dependents'].get(),
            'PhoneService': fields['PhoneService'].get(),
            'PaperlessBilling': fields['PaperlessBilling'].get(),
            'InternetService': fields['InternetService'].get(),
            'Contract': fields['Contract'].get(),
            'PaymentMethod': fields['PaymentMethod'].get(),
        }

        # Convert to DataFrame
        sample_df = pd.DataFrame([input_data])

        # Binary map
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            sample_df[col] = sample_df[col].map({'Yes': 1, 'No': 0})

        # One-hot encode and align
        sample_df = pd.get_dummies(sample_df)
        sample_df = sample_df.reindex(columns=feature_names, fill_value=0)

        # Predict
        scaled_input = scaler.transform(sample_df)
        prediction = model.predict(scaled_input)[0]
        result = "Yes (Customer likely to churn)" if prediction == 1 else "No (Customer likely to stay)"
        messagebox.showinfo("Prediction Result", f"Churn Prediction: {result}")

    except ValueError:
        messagebox.showerror("Input Error", "Please fill out all fields with valid numbers.")
    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

# Predict Button
predict_btn = tk.Button(root, text="Predict Churn", command=predict, bg="green", fg="white")
predict_btn.grid(row=11, column=0, columnspan=2, pady=20)

root.mainloop()
