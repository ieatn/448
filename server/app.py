from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np # Needed for numerical ops

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# --- Machine Learning Model Setup (Runs once when the app starts) ---

# 1. Load the simulated data
try:
    data = pd.read_csv('testing.csv')
    print("Successfully loaded 'testing.csv'")
except Exception as e:
    print(f"Error loading data: {e}")
    data = None 

# 2. Separate Features (X) and Target (y)
X = data.drop('HasDepression', axis=1)
y = data['HasDepression']

# 3. Define preprocessing steps
numerical_features = ['Age', 'Education_Years', 'Hours_Slept', 'Social_Activity_Score']
categorical_features = ['Occupation']
# Create preprocessor to handle numerical scaling and categorical encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. Create the Machine Learning Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])


# 5. Train the model
print("Training the machine learning model...")
model_pipeline.fit(X, y) # Train on the full dataset for simplicity in this example
print("Machine learning model trained and ready!")







# --- Flask Routes ---

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get data from the request
    # We expect the data to be sent as JSON in the request body
    data = request.get_json(force=True) # force=True tries to parse even if content-type isn't application/json
    print(f"Received data for prediction: {data}")

    # Basic validation: Check if all expected features are present
    expected_features = ['Age', 'Education_Years', 'Occupation', 'Hours_Slept', 'Social_Activity_Score']
    if not all(feature in data for feature in expected_features):
        return jsonify({"error": "Missing one or more required features in the input data."}), 400

    # 2. Convert received data into a Pandas DataFrame
    # This is crucial because our preprocessor and model expect a DataFrame-like structure
    try:
        input_df = pd.DataFrame([data]) # Wrap 'data' in a list to create a DataFrame with one row
    except Exception as e:
        return jsonify({"error": f"Failed to convert input data to DataFrame: {str(e)}"}), 400

    # 3. Make a prediction using the trained pipeline
    try:
        # The pipeline handles preprocessing automatically
        prediction_proba = model_pipeline.predict_proba(input_df)
        # prediction_proba will be like [[prob_no_depression, prob_depression]]

        # Get the probability of depression (class 1)
        depression_percentage = prediction_proba[0][1] * 100

        # Get the predicted class (0 or 1)
        predicted_class = int(model_pipeline.predict(input_df)[0]) # Convert numpy int to Python int

        # 4. Return the prediction as a JSON response
        response = {
            "predicted_depression": bool(predicted_class), # True/False
            "probability_of_depression": round(depression_percentage, 2) # Rounded to 2 decimal places
        }
        print(f"Prediction successful: {response}")
        return jsonify(response)

    except Exception as e:
        # Catch any errors during prediction (e.g., if input data format is unexpected)
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

