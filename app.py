import os
import pickle
from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd

app = Flask(__name__)

# =============================
# Load model & columns
# =============================

# Always look for model in same folder as app.py
MODEL_PATH = os.path.join(os.path.dirname(__file__), "decision_tree_model.pkl")

# ✅ Load model safely
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Model file not found at: {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Columns used for model input
model_columns = [
    'satisfaction_level',
    'last_evaluation',
    'number_project',
    'average_monthly_hours',
    'time_spend_company',
    'Work_accident',
    'promotion_last_5years',
    'high'
]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded. Please check your MODEL_PATH."})

        json_ = request.json
        if not json_:
            return jsonify({"error": "No input data received."})

        query_df = pd.DataFrame([json_])

        for col in model_columns:
            if col not in query_df.columns:
                query_df[col] = 0

        query_df = query_df[model_columns]
        prediction = model.predict(query_df)[0]
        result = "Employee likely to LEAVE" if prediction == 1 else "Employee likely to STAY"

        return jsonify({"prediction": int(prediction), "result": result})

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()})

if __name__ == "__main__":
    app.run(debug=True)
