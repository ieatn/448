from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import warnings

# Suppress pandas warnings
warnings.filterwarnings('ignore', category=FutureWarning)

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

# 4. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData Split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 5. Define preprocessing steps
categorical_features = ['Gender', 'Profession', 'Sleep Duration', 'Dietary Habits', 'Degree', 
                       'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# Create preprocessor to handle numerical scaling and categorical encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 6. Create the Machine Learning Pipeline with LASSO regularization
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        random_state=42,
        solver='liblinear',
        penalty='l1',  # LASSO regularization
        C=0.1,  # Inverse of regularization strength (smaller C = stronger regularization)
        max_iter=1000
    ))
])

# 7. Train the model
print("\nTraining model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete!")

# 8. Evaluate the model
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"Test Accuracy: {accuracy:.2f}")

# Print simplified classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred, output_dict=True)
print(f"Class 0 (No Depression):")
print(f"  Precision: {report['0']['precision']:.2f}")
print(f"  Recall: {report['0']['recall']:.2f}")
print(f"  F1-score: {report['0']['f1-score']:.2f}")
print(f"\nClass 1 (Depression):")
print(f"  Precision: {report['1']['precision']:.2f}")
print(f"  Recall: {report['1']['recall']:.2f}")
print(f"  F1-score: {report['1']['f1-score']:.2f}")

# Print feature importance from LASSO
print("\nFeature Importance (LASSO Coefficients):")
feature_names = numerical_features + [f"{cat}_{val}" for cat, vals in 
    zip(categorical_features, model_pipeline.named_steps['preprocessor']
        .named_transformers_['cat'].categories_) for val in vals]
coefficients = model_pipeline.named_steps['classifier'].coef_[0]
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})
# Sort by absolute coefficient value
feature_importance['Abs_Coefficient'] = abs(feature_importance['Coefficient'])
feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
print(feature_importance[['Feature', 'Coefficient']].head(10))

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

