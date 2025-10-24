import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained Decision Tree model
model = joblib.load('decision_tree_model.pkl')

# Define the columns that the model expects, in the correct order.
# This order has been updated to remove all department-related features.
model_columns = [
    'satisfaction_level', 'last_evaluation', 'number_project',
    'average_monthly_hours', 'time_spend_company', 'Work_accident',
    'promotion_last_5years', 'salary_high', 'salary_low', 'salary_medium'
]

@app.route('/')
def home():
    """Renders the main page of the application."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests from the web page."""
    try:
        # Get the form data from the user's request
        form_data = request.form.to_dict()

        # Extract numerical features
        satisfaction_level = float(form_data['satisfaction_level'])
        last_evaluation = float(form_data['last_evaluation'])
        number_project = int(form_data['number_project'])
        average_monthly_hours = int(form_data['average_monthly_hours'])
        time_spend_company = int(form_data['time_spend_company'])
        work_accident = int(form_data['work_accident'])
        promotion_last_5years = int(form_data['promotion_last_5years'])
        
        # Handle one-hot encoded categorical features
        salary = form_data['salary']
        
        # Create a dictionary to hold the features for prediction
        prediction_data = {col: 0 for col in model_columns}
        
        # Populate the dictionary with user-inputted values
        prediction_data['satisfaction_level'] = satisfaction_level
        prediction_data['last_evaluation'] = last_evaluation
        prediction_data['number_project'] = number_project
        prediction_data['average_monthly_hours'] = average_monthly_hours
        prediction_data['time_spend_company'] = time_spend_company
        prediction_data['Work_accident'] = work_accident
        prediction_data['promotion_last_5years'] = promotion_last_5years
        
        # Set the correct one-hot encoded columns to 1
        salary_col = f"salary_{salary}"
        if salary_col in prediction_data:
            prediction_data[salary_col] = 1
        
        # Convert the dictionary to a DataFrame in the correct order
        input_df = pd.DataFrame([prediction_data])
        input_df = input_df[model_columns]
        
        # Make a prediction
        prediction = model.predict(input_df)[0]
        
        # Format the result
        result = "likely to leave" if prediction == 1 else "likely to stay"
        
        return jsonify({'prediction': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # You may need to specify host='0.0.0.0' to make the app
    # accessible from outside the local machine in a production environment.
    app.run(debug=True)
