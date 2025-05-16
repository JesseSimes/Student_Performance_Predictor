import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Path Config
DATASET_PATH = r'F:\Projects\Student Acedemics Predictor\Dataset\Student_Performance.csv'
MODEL_PATH = r'F:\Projects\Student Acedemics Predictor\Model\student_performance_model.pkl'
SCALER_PATH = r'F:\Projects\Student Acedemics Predictor\Model\scaler.pkl'

# Feature_names
feature_names = [
    'Hours Studied', 
    'Previous Scores', 
    'Extracurricular Activities', 
    'Sleep Hours', 
    'Sample Question Papers Practiced'
]
numerical_features = [
    'Hours Studied', 
    'Previous Scores', 
    'Sleep Hours', 
    'Sample Question Papers Practiced'
]
categorical_features = ['Extracurricular Activities']
yes_no_map = {'yes': 1, 'no': 0}

# Loading clean data
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("Dataset not found at the specified path.")

df = pd.read_csv(DATASET_PATH)

df['Extracurricular Activities'] = df['Extracurricular Activities'].map({'Yes': 1, 'No': 0})

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

df.to_csv('student_cleaned.csv', index=False)

# Split and Train data
X = df[feature_names]
y = df['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n--- Model Evaluation ---")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE:  {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.2f}")
print(f"R2:   {r2_score(y_test, y_pred):.2f}")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
print("\nModel and Scaler saved successfully.")

# User Input
print("\n--- Student Performance Prediction ---")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("Model and Scaler loaded successfully.\n")

n = int(input("Enter number of students to predict for: "))
all_inputs = []

for i in range(n):
    print(f"\n--- Student {i+1} ---")
    student_input = []
    for feature in feature_names:
        if feature in categorical_features:
            val = input(f"Enter {feature} (yes/no): ").strip().lower()
            while val not in yes_no_map:
                val = input("Please enter 'yes' or 'no': ").strip().lower()
            student_input.append(yes_no_map[val])
        else:
            val = float(input(f"Enter {feature}: "))
            student_input.append(val)
    all_inputs.append(student_input)

# Convert to NumPy array
user_input_array = np.array(all_inputs)

# Scale numerical features only
user_input_scaled = user_input_array.copy()
num_indices = [feature_names.index(f) for f in numerical_features]
user_input_scaled[:, num_indices] = scaler.transform(user_input_array[:, num_indices])

# Predict
predictions = model.predict(user_input_scaled)

# Show results
print("\n--- Predicted Performance Indices ---")
for i, pred in enumerate(predictions, 1):
    print(f"Student {i}: {pred:.2f}")

# Optional: Plot actual vs predicted
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()
