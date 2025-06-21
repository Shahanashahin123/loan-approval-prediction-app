import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set the path to the dataset
data_path = os.path.join('data', 'loan_data.csv')
data = pd.read_csv(data_path)

print("Initial data shape:", data.shape)

# Clean column names (remove leading/trailing spaces)
data.columns = [col.strip() for col in data.columns]

# Strip strings from all object columns
data = data.apply(lambda col: col.str.strip() if col.dtype == "object" else col)


# Drop rows with nulls
data.dropna(inplace=True)
print("After dropping nulls:", data.shape)

# Encode categorical columns
data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})

# Feature and target
X = data.drop(['loan_id', 'loan_status'], axis=1)
# Convert 'Approved' to 1 and 'Rejected' to 0
data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})
y = data['loan_status']


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Accuracy: {accuracy}")

# Save model and scaler
model_path = os.path.join('backend', 'model.pkl')
scaler_path = os.path.join('backend', 'scaler.pkl')

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Model saved to {model_path}")
print(f"✅ Scaler saved to {scaler_path}")

