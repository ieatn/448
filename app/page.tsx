'use client';

import { useState } from 'react';
// Removed Image import as it's no longer used
// import Image from "next/image";

export default function Home() {
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [occupation, setOccupation] = useState('');
  const [education, setEducation] = useState('');
  const [income, setIncome] = useState('');
  const [exercise, setExercise] = useState('');
  const [social, setSocial] = useState('');
  const [sleep, setSleep] = useState('');
  const [prediction, setPrediction] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fillSampleData = () => {
    setAge('28');
    setGender('female');
    setOccupation('Software Engineer');
    setEducation('bachelors');
    setIncome('60k-100k');
    setExercise('3-4');
    setSocial('medium');
    setSleep('7.5');
  };
  


  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    const formData = { age, gender, occupation, education, income, exercise, social, sleep };

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const result = await response.json();
      setPrediction(result.prediction);

    } catch (err) {
      console.error('Failed to fetch prediction:', err);
      setError('Failed to get prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8 font-[family-name:var(--font-geist-sans)] dark:bg-gray-900 dark:text-white">
      <main className="flex flex-col gap-8 items-center w-full max-w-md bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <h1 className="text-2xl font-semibold text-center">Depression Risk Survey</h1>
        <form onSubmit={handleSubmit} className="w-full flex flex-col gap-4">
          {/* Age Input */}
          <div>
            <label htmlFor="age" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Age</label>
            <input
              type="number"
              id="age"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Gender Select */}
          <div>
            <label htmlFor="gender" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Gender</label>
            <select
              id="gender"
              value={gender}
              onChange={(e) => setGender(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Gender</option>
              <option value="male">Male</option>
              <option value="female">Female</option>
              <option value="non-binary">Non-binary</option>
              <option value="prefer-not-to-say">Prefer not to say</option>
            </select>
          </div>

          {/* Occupation Input */}
          <div>
            <label htmlFor="occupation" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Occupation</label>
            <input
              type="text"
              id="occupation"
              value={occupation}
              onChange={(e) => setOccupation(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Education Select */}
          <div>
            <label htmlFor="education" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Highest Education Level</label>
            <select
              id="education"
              value={education}
              onChange={(e) => setEducation(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Education</option>
              <option value="high-school">High School</option>
              <option value="bachelors">Bachelor's Degree</option>
              <option value="masters">Master's Degree</option>
              <option value="phd">PhD</option>
              <option value="other">Other</option>
            </select>
          </div>

          {/* Income Select */}
          <div>
            <label htmlFor="income" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Approximate Annual Income (USD)</label>
            <select
              id="income"
              value={income}
              onChange={(e) => setIncome(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Income Range</option>
              <option value="<30k">Less than $30,000</option>
              <option value="30k-60k">$30,000 - $59,999</option>
              <option value="60k-100k">$60,000 - $99,999</option>
              <option value="100k-150k">$100,000 - $149,999</option>
              <option value=">150k">$150,000 or more</option>
              <option value="prefer-not-to-say">Prefer not to say</option>
            </select>
          </div>

          {/* Exercise Frequency Select */}
          <div>
            <label htmlFor="exercise" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Exercise Frequency</label>
            <select
              id="exercise"
              value={exercise}
              onChange={(e) => setExercise(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Frequency</option>
              <option value="never">Never</option>
              <option value="1-2">1-2 times/week</option>
              <option value="3-4">3-4 times/week</option>
              <option value="5+">5+ times/week</option>
            </select>
          </div>

          {/* Social Interaction Select */}
          <div>
            <label htmlFor="social" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Social Interaction Level</label>
            <select
              id="social"
              value={social}
              onChange={(e) => setSocial(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Level</option>
              <option value="low">Low (Rarely interact)</option>
              <option value="medium">Medium (Few times a week)</option>
              <option value="high">High (Daily interactions)</option>
            </select>
          </div>

          {/* Sleep Hours Input */}
          <div>
            <label htmlFor="sleep" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Average Sleep Hours per Night</label>
            <input
              type="number"
              id="sleep"
              step="0.5"
              min="0" // Added min value
              max="24" // Added max value
              value={sleep}
              onChange={(e) => setSleep(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          <button
            type="button"
            onClick={fillSampleData}
            className="w-full inline-flex justify-center py-2 px-4 border border-indigo-600 shadow-sm text-sm font-medium rounded-md text-indigo-600 bg-white hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Fill with Sample Data
          </button>

          <button
            type="submit"
            disabled={isLoading} // Disable button while loading
            className="mt-4 w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Getting Prediction...' : 'Get Prediction'} {/* Show loading text */}
          </button>
        </form>

        {/* Display Error Message */}
        {error && (
          <div className="mt-6 p-4 border border-red-300 bg-red-50 dark:border-red-700 dark:bg-red-900 rounded-md w-full text-center">
            <p className="text-sm font-medium text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {/* Prediction results */}
        {prediction && (
          <div className="mt-6 p-4 border border-gray-300 dark:border-gray-700 rounded-md w-full text-center bg-gray-50 dark:bg-gray-700">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">Prediction Result:</h2>
            <p className="mt-2 text-gray-700 dark:text-gray-300">{prediction}</p>
            <p className="mt-4 text-xs text-gray-500 dark:text-gray-400">Disclaimer: This prediction is based on a statistical model and is not a substitute for professional medical advice. If you are concerned about your mental health, please consult a healthcare professional.</p>
          </div>
        )}

        {/* Tutorial Section */}
        <div className="mt-8 w-full">
          <h2 className="text-xl font-semibold mb-4">Tutorial: Working with Depression Data</h2>
          
          <div className="space-y-6">
            {/* Step 1: Data Loading */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Step 1: Loading the Data</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded overflow-x-auto text-sm">
{`import pandas as pd

# Load the depression scores data
df = pd.read_csv('depression_scores.csv')

# Display basic information
print(df.info())
print(df.describe())`}
              </pre>
            </div>

            {/* Step 2: Data Preprocessing */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Step 2: Data Preprocessing</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded overflow-x-auto text-sm">
{`# Handle missing values
df = df.fillna(df.mean())

# Convert categorical variables
df = pd.get_dummies(df, columns=['gender', 'education'])

# Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
numerical_cols = ['age', 'sleep', 'exercise']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])`}
              </pre>
            </div>

            {/* Step 3: Model Training */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Step 3: Model Training</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded overflow-x-auto text-sm">
{`from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Split the data
X = df.drop('depression_score', axis=1)
y = df['depression_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Model accuracy: {score:.2f}")`}
              </pre>
            </div>

            {/* Step 4: Making Predictions */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Step 4: Making Predictions</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded overflow-x-auto text-sm">
{`# Example of making a prediction
def predict_depression_risk(input_data):
    # Preprocess input data
    processed_data = preprocess_input(input_data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    # Return risk level
    risk_levels = {
        0: "Low Risk",
        1: "Moderate Risk",
        2: "High Risk"
    }
    return risk_levels[prediction[0]]`}
              </pre>
            </div>

            <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
              <p>Note: This is a simplified example. In a real application, you would need to:</p>
              <ul className="list-disc list-inside mt-2">
                <li>Implement proper error handling</li>
                <li>Add data validation</li>
                <li>Include model evaluation metrics</li>
                <li>Add cross-validation</li>
                <li>Implement proper model versioning</li>
                <li>Add logging and monitoring</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
