'use client';

import Link from 'next/link';
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
import { Bar, Radar, Line } from 'react-chartjs-2';
import { useState } from 'react';

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

export default function Analysis() {
  const [selectedMetric, setSelectedMetric] = useState<MetricName>('accuracy');
  const [selectedModel, setSelectedModel] = useState<ModelName>('K-Means');

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
        <div className="w-full flex justify-between items-center">
          <h1 className="text-2xl font-semibold">Depression Risk Model Analysis</h1>
          <Link 
            href="/" 
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Back to Survey
          </Link>
        </div>

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

        {/* Model Comparison Section */}
        <div className="w-full space-y-6">
          <h2 className="text-xl font-semibold">Model Comparison</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Bar Chart for Model Performance */}
            <div className="bg-white dark:bg-gray-700 p-4 rounded-lg shadow">
              <h3 className="text-lg font-medium mb-4">Performance Metrics</h3>
              <div>
                <select 
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value as MetricName)}
                  className="mb-4 pl-2 pr-8 py-1 text-base border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="accuracy">Accuracy</option>
                  <option value="training_time">Training Time</option>
                  <option value="interpretability">Interpretability</option>
                  <option value="scalability">Scalability</option>
                </select>
                <div className="h-[300px]">
                  <Bar options={barOptions} data={performanceData} />
                </div>
              </div>
            </div>

            {/* Radar Chart for Model Characteristics */}
            <div className="bg-white dark:bg-gray-700 p-4 rounded-lg shadow">
              <h3 className="text-lg font-medium mb-4">Model Characteristics</h3>
              <div>
                <select 
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value as ModelName)}
                  className="mb-4 pl-2 pr-8 py-1 text-base border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="K-Means">K-Means</option>
                  <option value="KNN">KNN</option>
                  <option value="LASSO">LASSO</option>
                  <option value="Linear Regression">Linear Regression</option>
                </select>
                <div className="h-[300px]">
                  <Radar options={radarOptions} data={radarData} />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tutorial Section */}
        <div className="w-full">
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