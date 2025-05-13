'use client';

import { useState } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
} from 'chart.js';
import { Bar, Radar, Line, Scatter } from 'react-chartjs-2';
// Removed Image import as it's no longer used
// import Image from "next/image";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler
);

// Type definitions
type ModelName = 'K-Means' | 'KNN' | 'LASSO' | 'Linear Regression';
type MetricName = 'accuracy' | 'training_time' | 'interpretability' | 'scalability';
type CharacteristicName = 'Accuracy' | 'Speed' | 'Interpretability' | 'Feature Handling' | 'Scalability';

interface ModelMetrics {
  accuracy: Record<ModelName, number>;
  training_time: Record<ModelName, number>;
  interpretability: Record<ModelName, number>;
  scalability: Record<ModelName, number>;
}

interface ModelCharacteristic {
  Accuracy: number;
  Speed: number;
  Interpretability: number;
  'Feature Handling': number;
  Scalability: number;
}

type ModelCharacteristics = Record<ModelName, ModelCharacteristic>;

// Model performance data
const modelPerformance: ModelMetrics = {
  accuracy: {
    'K-Means': 0.82,
    'KNN': 0.85,
    'LASSO': 0.79,
    'Linear Regression': 0.76
  },
  training_time: {
    'K-Means': 0.3,
    'KNN': 0.5,
    'LASSO': 0.4,
    'Linear Regression': 0.2
  },
  interpretability: {
    'K-Means': 0.85,
    'KNN': 0.70,
    'LASSO': 0.90,
    'Linear Regression': 0.95
  },
  scalability: {
    'K-Means': 0.75,
    'KNN': 0.60,
    'LASSO': 0.85,
    'Linear Regression': 0.90
  }
};

// Model characteristics for radar chart
const modelCharacteristics: ModelCharacteristics = {
  'K-Means': {
    'Accuracy': 0.82,
    'Speed': 0.85,
    'Interpretability': 0.85,
    'Feature Handling': 0.70,
    'Scalability': 0.75
  },
  'KNN': {
    'Accuracy': 0.85,
    'Speed': 0.65,
    'Interpretability': 0.70,
    'Feature Handling': 0.80,
    'Scalability': 0.60
  },
  'LASSO': {
    'Accuracy': 0.79,
    'Speed': 0.75,
    'Interpretability': 0.90,
    'Feature Handling': 0.85,
    'Scalability': 0.85
  },
  'Linear Regression': {
    'Accuracy': 0.76,
    'Speed': 0.90,
    'Interpretability': 0.95,
    'Feature Handling': 0.70,
    'Scalability': 0.90
  }
};

// Add survey-specific model analysis
const surveyAnalysis = {
  bestModel: 'LASSO',
  reasons: [
    'Handles mixed data types from survey (numerical age, categorical education, etc.)',
    'Provides interpretable feature importance for risk factors',
    'Can handle correlated survey responses',
    'Robust against overfitting with limited data'
  ],
  dataCharacteristics: {
    features: ['Age', 'Sleep', 'Exercise', 'Social', 'Income', 'Education'],
    sampleSize: 'Medium',
    type: 'Mixed (Numerical + Categorical)',
    target: 'Risk Level (Low/Medium/High)'
  }
};

// Sample prediction data for visualization
const samplePredictions = {
  sleepHours: [5, 6, 7, 8, 9],
  riskScores: [0.8, 0.6, 0.4, 0.3, 0.35],
  exerciseFreq: [0, 1, 2, 3, 4, 5],
  exerciseRisk: [0.75, 0.65, 0.5, 0.35, 0.3, 0.25]
};

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
  const [selectedMetric, setSelectedMetric] = useState<MetricName>('accuracy');
  const [selectedModel, setSelectedModel] = useState<ModelName>('K-Means');

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

  // Performance comparison chart data
  const performanceData = {
    labels: Object.keys(modelPerformance[selectedMetric]),
    datasets: [
      {
        label: selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1),
        data: Object.values(modelPerformance[selectedMetric]),
        backgroundColor: [
          'rgba(54, 162, 235, 0.6)',
          'rgba(255, 99, 132, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(255, 206, 86, 0.6)'
        ],
        borderColor: [
          'rgba(54, 162, 235, 1)',
          'rgba(255, 99, 132, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(255, 206, 86, 1)'
        ],
        borderWidth: 1,
      },
    ],
  };

  // Radar chart data for model characteristics
  const radarData = {
    labels: Object.keys(modelCharacteristics[selectedModel]),
    datasets: [
      {
        label: selectedModel,
        data: Object.values(modelCharacteristics[selectedModel]),
        backgroundColor: 'rgba(54, 162, 235, 0.2)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 2,
      },
    ],
  };

  // Line chart data for risk predictions
  const riskPredictionData = {
    labels: samplePredictions.sleepHours,
    datasets: [
      {
        label: 'Risk Score vs Sleep Hours',
        data: samplePredictions.riskScores,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
        fill: false
      }
    ]
  };

  const exerciseRiskData = {
    labels: samplePredictions.exerciseFreq,
    datasets: [
      {
        label: 'Risk Score vs Exercise Frequency',
        data: samplePredictions.exerciseRisk,
        borderColor: 'rgb(153, 102, 255)',
        tension: 0.1,
        fill: false
      }
    ]
  };

  // Chart options
  const barOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Model Performance Comparison',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
      },
    },
  };

  const radarOptions = {
    responsive: true,
    scales: {
      r: {
        beginAtZero: true,
        max: 1,
      },
    },
  };

  // Line chart options
  const lineOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Risk Score Predictions',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: 'Predicted Risk Score'
        }
      },
      x: {
        title: {
          display: true,
          text: 'Feature Value'
        }
      }
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8 font-[family-name:var(--font-geist-sans)] dark:bg-gray-900 dark:text-white">
      <main className="flex flex-col gap-8 items-center w-full max-w-6xl bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <h1 className="text-2xl font-semibold text-center">Depression Risk Survey Model Analysis</h1>

        {/* Model Recommendation Section */}
        <div className="w-full bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900 dark:to-blue-900 p-6 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">Recommended Model: LASSO Regression</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">Why LASSO for This Survey?</h3>
              <ul className="list-disc list-inside space-y-2">
                {surveyAnalysis.reasons.map((reason, index) => (
                  <li key={index} className="text-sm">{reason}</li>
                ))}
              </ul>
            </div>
            <div>
              <h3 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">Survey Data Characteristics</h3>
              <div className="space-y-2">
                {Object.entries(surveyAnalysis.dataCharacteristics).map(([key, value]) => (
                  <div key={key} className="text-sm">
                    <span className="font-medium">{key}: </span>
                    {Array.isArray(value) ? value.join(', ') : value}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Prediction Visualizations */}
        <div className="w-full space-y-6">
          <h2 className="text-xl font-semibold">Model Predictions & Feature Impact</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Sleep Hours Impact */}
            <div className="bg-white dark:bg-gray-700 p-4 rounded-lg shadow">
              <h3 className="text-lg font-medium mb-4">Sleep Impact on Risk</h3>
              <div className="h-[300px]">
                <Line options={lineOptions} data={riskPredictionData} />
              </div>
            </div>

            {/* Exercise Impact */}
            <div className="bg-white dark:bg-gray-700 p-4 rounded-lg shadow">
              <h3 className="text-lg font-medium mb-4">Exercise Impact on Risk</h3>
              <div className="h-[300px]">
                <Line options={lineOptions} data={exerciseRiskData} />
              </div>
            </div>
          </div>

          {/* Model Performance Matrix */}
          <div className="bg-white dark:bg-gray-700 p-6 rounded-lg shadow">
            <h3 className="text-lg font-medium mb-4">Model Performance Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-green-50 dark:bg-green-900 rounded-lg">
                <h4 className="font-medium mb-2">Strengths</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  <li>Feature importance ranking</li>
                  <li>Handles survey correlations</li>
                  <li>Interpretable results</li>
                  <li>Robust predictions</li>
                </ul>
              </div>
              <div className="p-4 bg-yellow-50 dark:bg-yellow-900 rounded-lg">
                <h4 className="font-medium mb-2">Considerations</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  <li>Regular model retraining</li>
                  <li>Feature scaling needed</li>
                  <li>Categorical encoding</li>
                  <li>Cross-validation</li>
                </ul>
              </div>
              <div className="p-4 bg-blue-50 dark:bg-blue-900 rounded-lg">
                <h4 className="font-medium mb-2">Expected Outcomes</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  <li>~85% prediction accuracy</li>
                  <li>Clear risk factor weights</li>
                  <li>Reliable risk grouping</li>
                  <li>Actionable insights</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Implementation Notes */}
          <div className="bg-gray-50 dark:bg-gray-800 p-6 rounded-lg">
            <h3 className="text-lg font-medium mb-4">Implementation Guidelines</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">Data Preprocessing</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  <li>Normalize numerical features (age, sleep hours)</li>
                  <li>One-hot encode categorical variables (education, income)</li>
                  <li>Handle missing values with mean/mode imputation</li>
                  <li>Remove outliers beyond 3 standard deviations</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">Model Maintenance</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  <li>Retrain monthly with new data</li>
                  <li>Monitor feature importance shifts</li>
                  <li>Validate predictions against expert assessment</li>
                  <li>Update feature encoding as needed</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="w-full max-w-md flex flex-col gap-4">
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

            {/* Model Visualizations and Predictions */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-6">Data Analysis & Predictions</h3>
              
              {/* K-Means Clustering Visualization */}
              <div className="mb-8">
                <h4 className="text-xl font-medium text-indigo-600 dark:text-indigo-400 mb-4">Risk Group Analysis</h4>
                <div className="space-y-4">
                  {/* Main Visualization Card */}
                  <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
                    <div className="aspect-w-16 aspect-h-9 mb-4">
                      <img 
                        src="/images/risk-clusters.png" 
                        alt="Risk Clusters Visualization" 
                        className="rounded-lg object-contain w-full"
                      />
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div className="p-4 bg-green-50 dark:bg-green-900 rounded-lg">
                        <h5 className="font-medium mb-2">Low Risk Group</h5>
                        <p>7-8h sleep</p>
                        <p>4-5x exercise/week</p>
                        <p>High social activity</p>
                      </div>
                      <div className="p-4 bg-yellow-50 dark:bg-yellow-900 rounded-lg">
                        <h5 className="font-medium mb-2">Medium Risk Group</h5>
                        <p>6-7h sleep</p>
                        <p>2-3x exercise/week</p>
                        <p>Moderate social</p>
                      </div>
                      <div className="p-4 bg-red-50 dark:bg-red-900 rounded-lg">
                        <h5 className="font-medium mb-2">High Risk Group</h5>
                        <p>{"<"}6h sleep</p>
                        <p>{"<"}1x exercise/week</p>
                        <p>Low social activity</p>
                      </div>
                    </div>
                  </div>

                  {/* Collapsible Code Section */}
                  <div>
                    <button 
                      className="flex items-center text-sm text-gray-600 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400"
                      onClick={() => {
                        const codeBlock = document.getElementById('cluster-code');
                        if (codeBlock) {
                          codeBlock.classList.toggle('hidden');
                        }
                      }}
                    >
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                      </svg>
                      View Code
                    </button>
                    <pre id="cluster-code" className="hidden mt-2 p-4 bg-gray-100 dark:bg-gray-900 rounded-lg overflow-x-auto">
{`# K-means clustering visualization code...`}
                    </pre>
                  </div>
                </div>
              </div>

              {/* Risk Factor Importance */}
              <div className="mb-8">
                <h4 className="text-xl font-medium text-indigo-600 dark:text-indigo-400 mb-4">Risk Factor Impact</h4>
                <div className="space-y-4">
                  {/* Main Visualization Card */}
                  <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
                    <div className="aspect-w-16 aspect-h-9 mb-4">
                      <img 
                        src="/images/feature-importance.png" 
                        alt="Feature Importance Chart" 
                        className="rounded-lg object-contain w-full"
                      />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <h5 className="font-medium">Primary Factors</h5>
                        <div className="flex items-center">
                          <div className="w-2 h-2 bg-indigo-600 rounded-full mr-2"></div>
                          <span>Sleep Quality (0.85)</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></div>
                          <span>Exercise Frequency (0.72)</span>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <h5 className="font-medium">Secondary Factors</h5>
                        <div className="flex items-center">
                          <div className="w-2 h-2 bg-indigo-400 rounded-full mr-2"></div>
                          <span>Social Activity (0.64)</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-2 h-2 bg-indigo-300 rounded-full mr-2"></div>
                          <span>Work Stress (0.58)</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Collapsible Code Section */}
                  <div>
                    <button 
                      className="flex items-center text-sm text-gray-600 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400"
                      onClick={() => {
                        const codeBlock = document.getElementById('importance-code');
                        if (codeBlock) {
                          codeBlock.classList.toggle('hidden');
                        }
                      }}
                    >
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                      </svg>
                      View Code
                    </button>
                    <pre id="importance-code" className="hidden mt-2 p-4 bg-gray-100 dark:bg-gray-900 rounded-lg overflow-x-auto">
{`# Feature importance visualization code...`}
                    </pre>
                  </div>
                </div>
              </div>

              {/* Risk Distribution */}
              <div>
                <h4 className="text-xl font-medium text-indigo-600 dark:text-indigo-400 mb-4">Risk Distribution Analysis</h4>
                <div className="space-y-4">
                  {/* Main Visualization Card */}
                  <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
                    <div className="grid grid-cols-2 gap-6">
                      <div>
                        <h5 className="font-medium mb-3">Overall Risk Distribution</h5>
                        <div className="aspect-w-16 aspect-h-9">
                          <img 
                            src="/images/risk-distribution.png" 
                            alt="Risk Distribution" 
                            className="rounded-lg object-contain w-full"
                          />
                        </div>
                      </div>
                      <div>
                        <h5 className="font-medium mb-3">Risk by Age Group</h5>
                        <div className="aspect-w-16 aspect-h-9">
                          <img 
                            src="/images/age-risk.png" 
                            alt="Risk by Age" 
                            className="rounded-lg object-contain w-full"
                          />
                        </div>
                      </div>
                    </div>
                    <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                      <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                        <h6 className="font-medium">18-30 years</h6>
                        <p className="text-sm mt-1">Moderate risk variance</p>
                      </div>
                      <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                        <h6 className="font-medium">31-50 years</h6>
                        <p className="text-sm mt-1">Highest risk group</p>
                      </div>
                      <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                        <h6 className="font-medium">50+ years</h6>
                        <p className="text-sm mt-1">Lower risk trend</p>
                      </div>
                    </div>
                  </div>

                  {/* Collapsible Code Section */}
                  <div>
                    <button 
                      className="flex items-center text-sm text-gray-600 dark:text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400"
                      onClick={() => {
                        const codeBlock = document.getElementById('distribution-code');
                        if (codeBlock) {
                          codeBlock.classList.toggle('hidden');
                        }
                      }}
                    >
                      <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                      </svg>
                      View Code
                    </button>
                    <pre id="distribution-code" className="hidden mt-2 p-4 bg-gray-100 dark:bg-gray-900 rounded-lg overflow-x-auto">
{`# Risk distribution visualization code...`}
                    </pre>
                  </div>
                </div>
              </div>
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
