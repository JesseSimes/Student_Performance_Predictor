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


### Error:
** ValueError: X has 5 features, but LinearRegression is expecting 6 features as input. **

---

### ✅ 1. **Fixed the Input Feature Mismatch (Bug Fix)**

#### **Problem:**

```
ValueError: X has 5 features, but LinearRegression is expecting 6 features as input.
```

#### **Cause:**

During training, the model was trained on **6 features**, but during prediction, only **5 were provided**.

#### **Fix:**

Included the **`Extracurricular Activities`** feature as a numeric value (`1` for yes, `0` for no) in the input, matching the training data.

---

### ✅ 2. **Used `pandas.DataFrame` for Prediction Input**

#### **Problem:**

You got warnings like:

```
UserWarning: X does not have valid feature names...
```

#### **Cause:**

You were passing a plain list/array to `scaler.transform()` and `model.predict()`, which lacked column names.

#### **Fix:**

Converted the user input to a `pandas.DataFrame` with **exact same column names** as used during training:

```python
input_dict = {
    'Hours Studied': [value],
    'Previous Scores': [value],
    'Extracurricular Activities': [0 or 1],
    'Sleep Hours': [value],
    'Sample Question Papers Practiced': [value]
}
user_input_df = pd.DataFrame(input_dict)
```

This ensures **feature alignment** and removes warnings.

---

### ✅ 3. **Refactored the Script for Clean Structure**

* Separated:

  * Model loading
  * Input collection
  * Preprocessing
  * Prediction
  * Output display

This makes it easier to read, maintain, and reuse.

---

### ✅ 4. **Visualized Actual vs Predicted Output**

You plotted a graph:

```python
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Performance Index")
plt.ylabel("Predicted Performance Index")
plt.title("Actual vs Predicted")
```

This confirmed the model's accuracy visually.

---

### ✅ 5. **Performance Evaluation Included**

You added metrics like:

* MAE (Mean Absolute Error)
* MSE
* RMSE
* R² (Coefficient of Determination)

These show the model is performing extremely well.

---

### 🔁 Summary of Key Additions/Fixes

| Change                                  | Purpose                                  |
| --------------------------------------- | ---------------------------------------- |
| Input data structure fixed (6 features) | Prevents shape mismatch error            |
| `pandas.DataFrame` with column names    | Removes feature name warnings            |
| Encoded `yes/no` as 1/0                 | Matches training format                  |
| Evaluation metrics                      | Confirms high accuracy                   |
| Plotting actual vs predicted            | Visual verification of model performance |

---
