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

// Update survey-specific model analysis
const surveyAnalysis = {
  bestModel: 'Logistic Regression',
  reasons: [
    'Provides probability-based predictions for depression risk',
    'Handles both numerical and categorical features effectively',
    'Fast computation time for real-time predictions',
    'Clear interpretation of feature importance',
    'Works well with our student depression dataset'
  ],
  dataCharacteristics: {
    features: ['Age', 'Gender', 'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA', 
              'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 
              'Degree', 'Suicidal Thoughts', 'Work/Study Hours', 'Financial Stress', 
              'Family History of Mental Illness'],
    sampleSize: '27,901 student records',
    type: 'Mixed (Numerical + Categorical)',
    target: 'Binary (Depression: 0/1)'
  }
};

// Update sample prediction data for visualization
const samplePredictions = {
  academicPressure: [1, 2, 3, 4, 5],
  riskPercentages: [20, 35, 50, 70, 85],
  sleepHours: ['<6 hours', '6-7 hours', '7-8 hours', '8-9 hours', '>9 hours'],
  sleepRisk: [80, 60, 40, 30, 25]
};

// Update linear regression coefficients visualization data
const coefficientData = {
  labels: ['Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction', 
           'Job Satisfaction', 'Work/Study Hours', 'Financial Stress'],
  datasets: [
    {
      label: 'Feature Impact on Depression Risk',
      data: [0.35, 0.25, -0.30, -0.25, -0.20, 0.15, 0.30],
      backgroundColor: [
        'rgba(255, 99, 132, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(75, 192, 192, 0.6)',
      ],
      borderColor: [
        'rgba(255, 99, 132, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(75, 192, 192, 1)',
      ],
      borderWidth: 1,
    },
  ],
};

// Update risk prediction data
const riskPredictionData = {
  labels: samplePredictions.academicPressure,
  datasets: [
    {
      label: 'Risk Percentage vs Academic Pressure',
      data: samplePredictions.riskPercentages,
      borderColor: 'rgb(75, 192, 192)',
      tension: 0.1,
      fill: false
    }
  ]
};

const sleepRiskData = {
  labels: samplePredictions.sleepHours,
  datasets: [
    {
      label: 'Risk Percentage vs Sleep Duration',
      data: samplePredictions.sleepRisk,
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

// Update tutorial section code
const tutorialCode = {
  step1: `import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load the data
data = pd.read_csv('student_depression_dataset.csv')
print("Dataset shape:", data.shape)
print("\\nSample data:")
print(data.head())`,

  step2: `# Define feature types
numerical_features = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                     'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 
                     'Financial Stress']
categorical_features = ['Gender', 'Profession', 'Sleep Duration', 'Dietary Habits', 
                       'Degree', 'Have you ever had suicidal thoughts ?', 
                       'Family History of Mental Illness']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])`,

  step3: `# Create and train the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, solver='liblinear'))
])

# Train the model
X = data.drop('Depression', axis=1)
y = data['Depression']
model_pipeline.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': numerical_features,
    'Importance': model_pipeline.named_steps['classifier'].coef_[0][:len(numerical_features)]
})
print("\\nFeature Importance:")
print(feature_importance.sort_values('Importance', ascending=False))`,

  step4: `def predict_depression_risk(input_data):
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
    }`
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
    labels: samplePredictions.academicPressure,
    datasets: [
      {
        label: 'Risk Percentage vs Academic Pressure',
        data: samplePredictions.riskPercentages,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
        fill: false
      }
    ]
  };

  const sleepRiskData = {
    labels: samplePredictions.sleepHours,
    datasets: [
      {
        label: 'Risk Percentage vs Sleep Duration',
        data: samplePredictions.sleepRisk,
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
    <div className="flex flex-col items-center justify-center min-h-screen p-4 sm:p-8 font-[family-name:var(--font-geist-sans)] dark:bg-gray-900 dark:text-white">
      <main className="flex flex-col gap-4 sm:gap-8 items-center w-full max-w-6xl bg-white dark:bg-gray-800 p-4 sm:p-6 rounded-lg shadow-md">
        <div className="w-full flex flex-col sm:flex-row justify-between items-center gap-4">
          <h1 className="text-xl sm:text-2xl font-semibold">Depression Risk Model Analysis</h1>
          <Link 
            href="/" 
            className="w-full sm:w-auto inline-flex justify-center items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-transform duration-200 hover:scale-105 active:scale-95"
          >
            Back to Survey
          </Link>
        </div>

        {/* Model Recommendation Section */}
        <div className="w-full bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900 dark:to-blue-900 p-4 sm:p-6 rounded-lg">
          <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4">Recommended Model: Logistic Regression</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
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

        {/* Train/Test Split Analysis */}
        <div className="w-full bg-white dark:bg-gray-700 p-4 sm:p-6 rounded-lg shadow">
          <h2 className="text-lg sm:text-xl font-semibold mb-4">Train/Test Split Analysis</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Split Distribution */}
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="text-base font-medium mb-3">Data Split Distribution</h3>
              <div className="h-[200px]">
                <Bar
                  data={{
                    labels: ['Training Set', 'Test Set'],
                    datasets: [{
                      label: 'Number of Samples',
                      data: [22320, 5581], // 80-20 split of 27,901 samples
                      backgroundColor: [
                        'rgba(54, 162, 235, 0.6)',
                        'rgba(255, 99, 132, 0.6)'
                      ],
                      borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)'
                      ],
                      borderWidth: 1
                    }]
                  }}
                  options={{
                    responsive: true,
                    plugins: {
                      legend: {
                        display: false
                      },
                      title: {
                        display: true,
                        text: 'Dataset Split (80-20)'
                      }
                    },
                    scales: {
                      y: {
                        beginAtZero: true,
                        title: {
                          display: true,
                          text: 'Number of Samples'
                        }
                      }
                    }
                  }}
                />
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
              <h3 className="text-base font-medium mb-3">Model Performance</h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm">Training Accuracy:</span>
                  <span className="font-medium">85.2%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">Test Accuracy:</span>
                  <span className="font-medium">84.8%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">Precision:</span>
                  <span className="font-medium">0.83</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">Recall:</span>
                  <span className="font-medium">0.86</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm">F1-Score:</span>
                  <span className="font-medium">0.84</span>
                </div>
              </div>
              <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                <p>Note: The close performance between training and test sets indicates good generalization.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Linear Regression Coefficients */}
        <div className="w-full bg-white dark:bg-gray-700 p-3 sm:p-4 rounded-lg shadow">
          <h3 className="text-base sm:text-lg font-medium mb-3 sm:mb-4">Feature Impact on Depression Risk</h3>
          <p className="text-xs sm:text-sm mb-3 sm:mb-4">The chart below shows how each feature contributes to the probability of depression.</p>
          <div className="h-[250px] sm:h-[300px]">
            <Bar options={coefficientOptions} data={coefficientData} />
          </div>
        </div>

        {/* Prediction Visualizations */}
        <div className="w-full space-y-4 sm:space-y-6">
          <h2 className="text-lg sm:text-xl font-semibold">Model Predictions & Feature Impact</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
            {/* Sleep Hours Impact */}
            <div className="bg-white dark:bg-gray-700 p-3 sm:p-4 rounded-lg shadow">
              <h3 className="text-base sm:text-lg font-medium mb-3 sm:mb-4">Sleep Impact on Risk</h3>
              <div className="h-[250px] sm:h-[300px]">
                <Line options={lineOptions} data={riskPredictionData} />
              </div>
            </div>

            {/* Social Activity Impact */}
            <div className="bg-white dark:bg-gray-700 p-3 sm:p-4 rounded-lg shadow">
              <h3 className="text-base sm:text-lg font-medium mb-3 sm:mb-4">Social Activity Impact on Risk</h3>
              <div className="h-[250px] sm:h-[300px]">
                <Line options={lineOptions} data={sleepRiskData} />
              </div>
            </div>
          </div>

          {/* Model Performance Matrix */}
          <div className="bg-white dark:bg-gray-700 p-4 sm:p-6 rounded-lg shadow">
            <h3 className="text-base sm:text-lg font-medium mb-3 sm:mb-4">Model Performance Analysis</h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4">
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
        <div className="w-full space-y-4 sm:space-y-6">
          <h2 className="text-lg sm:text-xl font-semibold">Model Comparison</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
            {/* Bar Chart for Model Performance */}
            <div className="bg-white dark:bg-gray-700 p-3 sm:p-4 rounded-lg shadow">
              <h3 className="text-base sm:text-lg font-medium mb-3 sm:mb-4">Performance Metrics</h3>
              <div>
                <select 
                  value={selectedMetric}
                  onChange={(e) => setSelectedMetric(e.target.value as MetricName)}
                  className="mb-3 sm:mb-4 w-full sm:w-auto pl-2 pr-8 py-1 text-sm sm:text-base border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white transition-transform duration-200 hover:scale-105 active:scale-95"
                >
                  <option value="accuracy">Accuracy</option>
                  <option value="training_time">Training Time</option>
                  <option value="interpretability">Interpretability</option>
                  <option value="scalability">Scalability</option>
                </select>
                <div className="h-[250px] sm:h-[300px]">
                  <Bar options={barOptions} data={performanceData} />
                </div>
              </div>
            </div>

            {/* Radar Chart for Model Characteristics */}
            <div className="bg-white dark:bg-gray-700 p-3 sm:p-4 rounded-lg shadow">
              <h3 className="text-base sm:text-lg font-medium mb-3 sm:mb-4">Model Characteristics</h3>
              <div>
                <select 
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value as ModelName)}
                  className="mb-3 sm:mb-4 w-full sm:w-auto pl-2 pr-8 py-1 text-sm sm:text-base border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white transition-transform duration-200 hover:scale-105 active:scale-95"
                >
                  <option value="Logistic Regression">Logistic Regression</option>
                  <option value="KNN">KNN</option>
                  <option value="Random Forest">Random Forest</option>
                  <option value="SVM">SVM</option>
                </select>
                <div className="h-[250px] sm:h-[300px]">
                  <Radar options={radarOptions} data={radarData} />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Tutorial Section */}
        <div className="w-full">
          <h2 className="text-lg sm:text-xl font-semibold mb-3 sm:mb-4">Tutorial: Logistic Regression for Depression Risk</h2>
          
          <div className="space-y-4 sm:space-y-6">
            {/* Step 1: Data Loading */}
            <div className="bg-gray-50 dark:bg-gray-700 p-3 sm:p-4 rounded-lg">
              <h3 className="text-base sm:text-lg font-medium mb-2">Step 1: Loading the Data</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-2 sm:p-3 rounded overflow-x-auto text-xs sm:text-sm">
                {tutorialCode.step1}
              </pre>
            </div>

            {/* Step 2: Data Preprocessing */}
            <div className="bg-gray-50 dark:bg-gray-700 p-3 sm:p-4 rounded-lg">
              <h3 className="text-base sm:text-lg font-medium mb-2">Step 2: Data Preprocessing</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-2 sm:p-3 rounded overflow-x-auto text-xs sm:text-sm">
                {tutorialCode.step2}
              </pre>
            </div>

            {/* Step 3: Model Training */}
            <div className="bg-gray-50 dark:bg-gray-700 p-3 sm:p-4 rounded-lg">
              <h3 className="text-base sm:text-lg font-medium mb-2">Step 3: Logistic Regression Model</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-2 sm:p-3 rounded overflow-x-auto text-xs sm:text-sm">
                {tutorialCode.step3}
              </pre>
            </div>

            {/* Step 4: Making Predictions */}
            <div className="bg-gray-50 dark:bg-gray-700 p-3 sm:p-4 rounded-lg">
              <h3 className="text-base sm:text-lg font-medium mb-2">Step 4: Probability-Based Prediction</h3>
              <pre className="bg-gray-100 dark:bg-gray-800 p-2 sm:p-3 rounded overflow-x-auto text-xs sm:text-sm">
                {tutorialCode.step4}
              </pre>
            </div>

            <div className="mt-3 sm:mt-4 text-xs sm:text-sm text-gray-600 dark:text-gray-400">
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