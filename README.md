# 🎓 Student Performance Predictor

This project uses a Linear Regression model to predict a student's academic performance index based on various influencing factors. The goal is to build a simple and interpretable machine-learning pipeline that allows for efficient analysis and prediction from structured data.

---

## 📌 Problem Statement

We want to predict a student's **Performance Index** using features such as:
- Hours Studied
- Previous Scores
- Extracurricular Activities
- Sleep Hours
- Sample Question Papers Practiced

---

## 🧠 Skills Learned

Throughout this project, I practised and learned:

### 🔍 Data Discovery
- Using `os` to locate files in the local system
- Importing datasets with `pandas`
- Checking structure with `df.info()`, `df.head()`

### 📉 Exploratory Data Analysis (EDA)
- Identified missing values, duplicated rows, and datatype mismatches
- Used `.isnull()`, `.duplicated()`, `.value_counts()`
- Ensured all numeric columns were suitable for modelling

### 🧼 Data Preprocessing
- Mapped categorical column (`Extracurricular`: Yes/No → 1/0)
- Converted object datatypes to numeric (`float64`, `int64`)
- Applied **StandardScaler** normalization to remove scale bias
- Created a checklist:
  - ✅ No missing or broken data
  - ✅ No duplicates
  - ✅ Categorical values encoded
  - ✅ Normalized features
  - ✅ Target value isolated

### 🧪 Train/Test Splitting
- Used `train_test_split` with `random_state=42` to ensure reproducibility
- 80/20 split to simulate real-world unseen data testing
- Learned **why splitting is essential** even when data is structured

### 📈 Model Training
I chose **Linear Regression** because it's interpretable and well-suited for structured numeric datasets
- Trained using only numerical features after normalization

### ⚠️ Common Errors & Fixes
- **ValueError: Could not convert string to float**
  - Solved by mapping `'Yes'/'No'` to `1/0` in `Extracurricular` column
- Resolved `dtype: object` issues by ensuring all input features are numeric

### 📊 Model Evaluation
- Calculated performance metrics:
  - **R² Score**
  - **Mean Squared Error (MSE)**
- Interpreted regression coefficients using `.coef_` and `.intercept_`

### 💾 Bonus (Optional Skills)
- Learned how to **save** the model using `joblib` for future use:
  ```python
  import joblib
  joblib.dump(model, 'student_performance_model.pkl')

### Model Accuracy:  
**Mean Absolute Error:** 1.6111213463123035

**Mean Squared Error:** 4.082628398521851

**Root Mean Squared Error:** 2.020551508505005

**R^2 Score: 98.89832909573145%**


