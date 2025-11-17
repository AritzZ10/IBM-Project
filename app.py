import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------
# 1️⃣ TRAIN MODEL (Only if not trained earlier)
# -------------------------------------------------------

MODEL_PATH = "models/salary_model.pkl"
DATA_PATH = "Salary Data.csv"

if not os.path.exists(MODEL_PATH):

    print("\n🔵 Training model...")

    # Load dataset
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=["Salary"])

    # Features and target
    X = data.drop(columns=["Salary"])
    y = data["Salary"]

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    # Preprocessing
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    # Model pipeline
    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42))
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.2f}, R2: {r2:.2f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("✅ Model trained & saved!")

else:
    print("📦 Model already exists — skipping training.")


# -------------------------------------------------------
# 2️⃣ LOAD MODEL FOR API
# -------------------------------------------------------

model = joblib.load(MODEL_PATH)


# -------------------------------------------------------
# 3️⃣ FLASK API
# -------------------------------------------------------

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form inputs
        age = float(request.form['age'])
        job_title = request.form['job_title']
        education_level = request.form['education_level']
        gender = request.form['gender']
        years_exp = float(request.form['years_experience'])

        # Prepare DataFrame for prediction
        input_data = pd.DataFrame([{
            'Age': age,
            'Job Title': job_title,
            'Education Level': education_level,
            'Gender': gender,
            'Years of Experience': years_exp
        }])

        # Predict
        salary = model.predict(input_data)[0]

        return render_template("index.html",
                               prediction_text=f"Predicted Salary: ₹{salary:,.2f}")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
