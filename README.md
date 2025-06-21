# 💸 Loan Approval Prediction App

This is a machine learning web application that predicts whether a loan application will be **Approved ✅** or **Rejected ❌** based on the user's input.

## 🔍 Project Overview

This project uses a trained ML model to analyze user-submitted data and determine the loan approval outcome. It includes:
- 🧠 Model training with historical loan data
- 🖥️ Backend built using **Flask**
- 🌐 A simple, styled frontend UI
- 🧪 Scikit-learn, Pandas, NumPy for ML preprocessing and model building

---

## 🚀 Features

- ✅ Predicts loan approval instantly
- 📊 Uses real dataset (`loan_data.csv`)
- 📦 Model saved as `model.pkl`, scaler as `scaler.pkl`
- 🎨 Colorful UI with background styling
- 🔐 Ignores `venv/` and other unnecessary files using `.gitignore`

---

## 🏗️ Project Structure

loan-approval-prediction-app/
├── backend/
│ ├── app.py
│ ├── train_model.py
│ ├── model.pkl
│ ├── scaler.pkl
├── frontend/
│ ├── index.html
│ └── styles.css
├── data/
│ └── loan_data.csv
├── .gitignore
├── requirements.txt
└── README.md

---

## 🛠️ How to Run the App

### 1. 🔧 Setup Environment

python -m venv venv
venv\Scripts\activate   # For Windows
pip install -r requirements.txt

##2. 📈 Train the Model

python backend/train_model.py
This will generate:

model.pkl

scaler.pkl

3. 🚀 Run the App

python backend/app.py
Then open your browser and go to:

👉 http://localhost:8000

## 🖼️ Screenshots

### Home Page
![Home Page](s![Screenshot 2025-06-21 194233](https://github.com/user-attachments/assets/91cf66e2-17b9-4b8d-ae87-e10dc404605a)
)

### Prediction Result
![Prediction Result](![Screenshot 2025-06-21 194203](https://github.com/user-attachments/assets/8a078e6f-b472-4238-8ad9-039a56903a9d)
)


