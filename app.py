from flask import Flask, request, jsonify, render_template
import traceback
import pandas as pd
import pickle

# =============================
# Load model & columns
# =============================

MODEL_PATH = r"D:\Data Science Files\Practice\ML end to end practice project\decision_tree_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

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

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# =============================
# Prediction Endpoint
# =============================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        json_ = request.json

        query_df = pd.DataFrame([json_])

        for col in model_columns:
            if col not in query_df.columns:
                query_df[col] = 0

        query_df = query_df[model_columns]

        prediction = model.predict(query_df)[0]

        result = "Employee likely to LEAVE" if prediction == 1 else "Employee likely to STAY"

        return jsonify({
            "prediction": int(prediction),
            "result": result
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        })

if __name__ == "__main__":
    app.run(debug=True)
