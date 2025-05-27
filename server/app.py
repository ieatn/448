from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Machine Learning Model Setup (Runs once when the app starts) ---

# 1. Load the new dataset
try:
    data = pd.read_csv('student_depression_dataset.csv')
    print("Successfully loaded 'student_depression_dataset.csv'")
except Exception as e:
    print(f"Error loading data: {e}")
    data = None 

# 2. Clean the data
# Replace '?' with NaN and convert numerical columns to float
numerical_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 
                     'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
for feature in numerical_features:
    data[feature] = pd.to_numeric(data[feature].replace('?', np.nan), errors='coerce')

# Fill missing values with median for numerical features
for feature in numerical_features:
    data[feature].fillna(data[feature].median(), inplace=True)

# 3. Separate Features (X) and Target (y)
X = data.drop('Depression', axis=1)
y = data['Depression']

# 4. Define preprocessing steps
categorical_features = ['Gender', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree', 
                       'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# Create preprocessor to handle numerical scaling and categorical encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 5. Create the Machine Learning Pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# 6. Train the model
print("Training the machine learning model...")
model_pipeline.fit(X, y)
print("Machine learning model trained and ready!")

# --- Flask Routes ---

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Get data from the request
    data = request.get_json(force=True)
    print(f"Received data for prediction: {data}")

    # Basic validation: Check if all expected features are present
    expected_features = ['Age', 'Gender', 'Profession', 'Academic Pressure', 'Work Pressure', 
                        'CGPA', 'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 
                        'Dietary Habits', 'Degree', 'Have you ever had suicidal thoughts ?', 
                        'Work/Study Hours', 'Financial Stress', 'Family History of Mental Illness']
    
    if not all(feature in data for feature in expected_features):
        return jsonify({"error": "Missing one or more required features in the input data."}), 400

    # 2. Convert received data into a Pandas DataFrame
    try:
        input_df = pd.DataFrame([data])
    except Exception as e:
        return jsonify({"error": f"Failed to convert input data to DataFrame: {str(e)}"}), 400

    # 3. Make a prediction using the trained pipeline
    try:
        prediction_proba = model_pipeline.predict_proba(input_df)
        depression_percentage = prediction_proba[0][1] * 100
        predicted_class = int(model_pipeline.predict(input_df)[0])

        # 4. Return the prediction as a JSON response
        response = {
            "predicted_depression": bool(predicted_class),
            "probability_of_depression": round(depression_percentage, 2)
        }
        print(f"Prediction successful: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)

