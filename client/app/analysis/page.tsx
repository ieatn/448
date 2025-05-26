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
type ModelName = 'Logistic Regression' | 'KNN' | 'Random Forest' | 'SVM';
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
    'Logistic Regression': 0.85,
    'KNN': 0.82,
    'Random Forest': 0.83,
    'SVM': 0.81
  },
  training_time: {
    'Logistic Regression': 0.2,
    'KNN': 0.5,
    'Random Forest': 0.8,
    'SVM': 0.7
  },
  interpretability: {
    'Logistic Regression': 0.95,
    'KNN': 0.70,
    'Random Forest': 0.75,
    'SVM': 0.65
  },
  scalability: {
    'Logistic Regression': 0.90,
    'KNN': 0.60,
    'Random Forest': 0.85,
    'SVM': 0.80
  }
};

// Model characteristics for radar chart
const modelCharacteristics: ModelCharacteristics = {
  'Logistic Regression': {
    'Accuracy': 0.85,
    'Speed': 0.90,
    'Interpretability': 0.95,
    'Feature Handling': 0.85,
    'Scalability': 0.90
  },
  'KNN': {
    'Accuracy': 0.82,
    'Speed': 0.65,
    'Interpretability': 0.70,
    'Feature Handling': 0.80,
    'Scalability': 0.60
  },
  'Random Forest': {
    'Accuracy': 0.83,
    'Speed': 0.75,
    'Interpretability': 0.75,
    'Feature Handling': 0.90,
    'Scalability': 0.85
  },
  'SVM': {
    'Accuracy': 0.81,
    'Speed': 0.70,
    'Interpretability': 0.65,
    'Feature Handling': 0.75,
    'Scalability': 0.80
  }
};

// Add survey-specific model analysis
const surveyAnalysis = {
  bestModel: 'Logistic Regression',
  reasons: [
    'Provides probability-based predictions for depression risk',
    'Handles both numerical and categorical features effectively',
    'Fast computation time for real-time predictions',
    'Clear interpretation of feature importance',
    'Works well with our balanced dataset of 30 samples'
  ],
  dataCharacteristics: {
    features: ['Age', 'Education_Years', 'Occupation', 'Hours_Slept', 'Social_Activity_Score'],
    sampleSize: '30 samples',
    type: 'Mixed (Numerical + Categorical)',
    target: 'Binary (HasDepression: 0/1)'
  }
};

// Sample prediction data for visualization
const samplePredictions = {
  sleepHours: [4, 5, 6, 7, 8, 9],
  riskPercentages: [85, 75, 60, 40, 25, 20],
  socialScores: [1, 3, 5, 7, 9, 10],
  socialRisk: [80, 65, 50, 35, 20, 15]
};

// Linear regression coefficients visualization data
const coefficientData = {
  labels: ['Age', 'Education_Years', 'Hours_Slept', 'Social_Activity_Score'],
  datasets: [
    {
      label: 'Feature Impact on Depression Risk',
      data: [0.15, -0.25, -0.35, -0.30],
      backgroundColor: [
        'rgba(255, 99, 132, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(75, 192, 192, 0.6)',
      ],
      borderColor: [
        'rgba(255, 99, 132, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(75, 192, 192, 1)',
      ],
      borderWidth: 1,
    },
  ],
};

export default function Analysis() {
  const [selectedMetric, setSelectedMetric] = useState<MetricName>('accuracy');
  const [selectedModel, setSelectedModel] = useState<ModelName>('Logistic Regression');

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
        label: 'Risk Percentage vs Sleep Hours',
        data: samplePredictions.riskPercentages,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
        fill: false
      }
    ]
  };

  const socialRiskData = {
    labels: samplePredictions.socialScores,
    datasets: [
      {
        label: 'Risk Percentage vs Social Activity Score',
        data: samplePredictions.socialRisk,
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

  const coefficientOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Feature Impact on Depression Risk',
      },
    },
    scales: {
      y: {
        beginAtZero: false,
        title: {
          display: true,
          text: 'Impact on Risk'
        }
      }
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
        text: 'Risk Percentage Predictions',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: 'Predicted Risk Percentage'
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
          <h2 className="text-xl font-semibold mb-4">Recommended Model: Logistic Regression</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">Why Logistic Regression for This Survey?</h3>
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

        {/* Linear Regression Coefficients */}
        <div className="w-full bg-white dark:bg-gray-700 p-4 rounded-lg shadow">
          <h3 className="text-lg font-medium mb-4">Feature Impact on Depression Risk</h3>
          <p className="text-sm mb-4">The chart below shows how each feature contributes to the probability of depression.</p>
          <div className="h-[300px]">
            <Bar options={coefficientOptions} data={coefficientData} />
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

            {/* Social Activity Impact */}
            <div className="bg-white dark:bg-gray-700 p-4 rounded-lg shadow">
              <h3 className="text-lg font-medium mb-4">Social Activity Impact on Risk</h3>
              <div className="h-[300px]">
                <Line options={lineOptions} data={socialRiskData} />
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
                  <li>Handles both numerical and categorical features</li>
                  <li>Provides probability-based predictions</li>
                  <li>Fast computation time</li>
                  <li>Clear interpretation of feature importance</li>
                </ul>
              </div>
              <div className="p-4 bg-yellow-50 dark:bg-yellow-900 rounded-lg">
                <h4 className="font-medium mb-2">Considerations</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  <li>Assumes linear relationships</li>
                  <li>Feature scaling important</li>
                  <li>Categorical encoding needed</li>
                  <li>Sensitive to outliers</li>
                </ul>
              </div>
              <div className="p-4 bg-blue-50 dark:bg-blue-900 rounded-lg">
                <h4 className="font-medium mb-2">Expected Outcomes</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  <li>Precise probability-based predictions</li>
                  <li>Clear feature importance ranking</li>
                  <li>Transparent prediction logic</li>
                  <li>Actionable risk insights</li>
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
                  <li>One-hot encode categorical variables</li>
                  <li>Handle missing values with mean/mode imputation</li>
                  <li>Consider feature interactions</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">Model Maintenance</h4>
                <ul className="list-disc list-inside text-sm space-y-1">
                  <li>Recalibrate coefficients as needed</li>
                  <li>Monitor prediction distribution</li>
                  <li>Compare with expert assessments</li>
                  <li>Adjust base risk as population changes</li>
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
                  <option value="Logistic Regression">Logistic Regression</option>
                  <option value="KNN">KNN</option>
                  <option value="Random Forest">Random Forest</option>
                  <option value="SVM">SVM</option>
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
          <h2 className="text-xl font-semibold mb-4">Tutorial: Logistic Regression for Depression Risk</h2>
          
          <div className="space-y-6">
            {/* Step 1: Data Loading */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Step 1: Loading the Data</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded overflow-x-auto text-sm">
{`import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv('testing.csv')
print("Dataset shape:", data.shape)
print("\\nSample data:")
print(data.head())`}
              </pre>
            </div>

            {/* Step 2: Data Preprocessing */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Step 2: Data Preprocessing</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded overflow-x-auto text-sm">
{`# Define feature types
numerical_features = ['Age', 'Education_Years', 'Hours_Slept', 'Social_Activity_Score']
categorical_features = ['Occupation']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])`}
              </pre>
            </div>

            {/* Step 3: Model Training */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Step 3: Logistic Regression Model</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded overflow-x-auto text-sm">
{`# Create and train the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# Train the model
X = data.drop('HasDepression', axis=1)
y = data['HasDepression']
model_pipeline.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': numerical_features,
    'Importance': model_pipeline.named_steps['classifier'].coef_[0][:len(numerical_features)]
})
print("\\nFeature Importance:")
print(feature_importance.sort_values('Importance', ascending=False))`}
              </pre>
            </div>

            {/* Step 4: Making Predictions */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Step 4: Probability-Based Prediction</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-3 rounded overflow-x-auto text-sm">
{`def predict_depression_risk(input_data):
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Get prediction probabilities
    proba = model_pipeline.predict_proba(input_df)
    depression_prob = proba[0][1] * 100
    
    # Get binary prediction
    prediction = model_pipeline.predict(input_df)[0]
    
    return {
        "predicted_depression": bool(prediction),
        "probability_of_depression": round(depression_prob, 2)
    }`}
              </pre>
            </div>

            <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
              <p>Note: This logistic regression implementation has advantages over categorical approaches:</p>
              <ul className="list-disc list-inside mt-2">
                <li>Provides probability-based predictions</li>
                <li>Shows exact feature impact through coefficients</li>
                <li>Allows for more personalized risk assessment</li>
                <li>Risk changes linearly with feature changes, easier to interpret</li>
                <li>Can be easily visualized with probability bars</li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
} 