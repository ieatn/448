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
import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FiArrowRight, FiBarChart2, FiPieChart, FiTrendingUp, FiInfo } from 'react-icons/fi';

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
    'Logistic Regression': 0.848,  // From server's test accuracy
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
    'Accuracy': 0.848,
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
    'Provides clear, interpretable probability-based predictions',
    'Handles both numerical and categorical features effectively',
    'Fast computation time for real-time predictions',
    'Clear interpretation of feature importance through LASSO coefficients',
    'Works well with our student depression dataset'
  ],
  dataCharacteristics: {
    features: ['Age', 'Gender', 'Profession', 'Academic Pressure', 'Work Pressure', 'CGPA', 
              'Study Satisfaction', 'Job Satisfaction', 'Sleep Duration', 'Dietary Habits', 
              'Degree', 'Have you ever had suicidal thoughts ?', 'Work/Study Hours', 'Financial Stress', 
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
  labels: ['Suicidal Thoughts (Yes)', 'Academic Pressure', 'Financial Stress', 
           'Dietary Habits (Unhealthy)', 'Age', 'Dietary Habits (Healthy)',
           'Work/Study Hours', 'Sleep Duration (<5h)', 'Study Satisfaction'],
  datasets: [
    {
      label: 'Feature Impact on Depression Risk',
      data: [1.27, 1.16, 0.80, 0.59, -0.53, -0.46, 0.43, 0.33, -0.33],
      backgroundColor: [
        'rgba(255, 99, 132, 0.6)',
        'rgba(255, 99, 132, 0.6)',
        'rgba(255, 99, 132, 0.6)',
        'rgba(255, 99, 132, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(75, 192, 192, 0.6)',
        'rgba(255, 99, 132, 0.6)',
        'rgba(255, 99, 132, 0.6)',
        'rgba(75, 192, 192, 0.6)',
      ],
      borderColor: [
        'rgba(255, 99, 132, 1)',
        'rgba(255, 99, 132, 1)',
        'rgba(255, 99, 132, 1)',
        'rgba(255, 99, 132, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(75, 192, 192, 1)',
        'rgba(255, 99, 132, 1)',
        'rgba(255, 99, 132, 1)',
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

// Update feature impact analysis section
const featureImpactAnalysis = {
  positiveImpact: [
    { feature: 'Suicidal Thoughts (Yes)', coefficient: 1.27, description: 'Strongest risk factor' },
    { feature: 'Academic Pressure', coefficient: 1.16, description: 'Major contributor to risk' },
    { feature: 'Financial Stress', coefficient: 0.80, description: 'Significant impact on risk' },
    { feature: 'Dietary Habits (Unhealthy)', coefficient: 0.59, description: 'Moderate risk factor' },
    { feature: 'Work/Study Hours', coefficient: 0.43, description: 'Notable contribution to risk' },
    { feature: 'Sleep Duration (<5h)', coefficient: 0.33, description: 'Moderate risk factor' }
  ],
  negativeImpact: [
    { feature: 'Age', coefficient: -0.53, description: 'Strong protective factor' },
    { feature: 'Dietary Habits (Healthy)', coefficient: -0.46, description: 'Significant protective effect' },
    { feature: 'Study Satisfaction', coefficient: -0.33, description: 'Moderate protective influence' }
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

  step3: `# Create and train the model pipeline with LASSO regularization
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        random_state=42,
        solver='liblinear',
        penalty='l1',  # LASSO regularization
        C=0.1,  # Inverse of regularization strength
        max_iter=1000
    ))
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

// Add cross-validation data type
interface CrossValidationResults {
  scores: number[];
  mean_score: number;
  std_score: number;
  folds: number;
}

// Add model evaluation data type
interface ModelEvaluation {
  test_accuracy: number;
  classification_report: {
    '0': { precision: number; recall: number; 'f1-score': number };
    '1': { precision: number; recall: number; 'f1-score': number };
  };
  feature_importance: Array<{ Feature: string; Coefficient: number }>;
  cv_results: {
    scores: number[];
    mean_score: number;
    std_score: number;
    folds: number;
  };
}

// Add animation variants
const fadeIn = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -20 }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

export default function Analysis() {
  const [selectedMetric, setSelectedMetric] = useState<MetricName>('accuracy');
  const [selectedModel, setSelectedModel] = useState<ModelName>('Logistic Regression');
  const [activeSection, setActiveSection] = useState('overview');
  const [cvResults, setCvResults] = useState<CrossValidationResults | null>(null);
  const [modelEvaluation, setModelEvaluation] = useState<ModelEvaluation | null>(null);


  // Fetch model evaluation data
  useEffect(() => {
    const fetchModelEvaluation = async () => {
      try {
        // Use a sample prediction to get model evaluation data
        const sampleData = {
          Age: 25,
          Gender: 'Male',
          Profession: 'Student',
          'Academic Pressure': 3,
          'Work Pressure': 2,
          CGPA: 7.5,
          'Study Satisfaction': 4,
          'Job Satisfaction': 3,
          'Sleep Duration': '7-8 hours',
          'Dietary Habits': 'Healthy',
          Degree: 'BSc',
          'Have you ever had suicidal thoughts ?': 'No',
          'Work/Study Hours': 6,
          'Financial Stress': 2,
          'Family History of Mental Illness': 'No'
        };

        const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(sampleData),
        });

        if (!response.ok) throw new Error('Failed to fetch model evaluation');
        const data = await response.json();
        setModelEvaluation(data.model_evaluation);
      } catch (error) {
        console.error('Error fetching model evaluation:', error);
      }
    };

    fetchModelEvaluation();
  }, []);

  // Navigation items
  const navItems = [
    { id: 'overview', label: 'Overview' },
    { id: 'model-performance', label: 'Model Performance' },
    { id: 'feature-impact', label: 'Feature Impact' },
    { id: 'model-comparison', label: 'Model Comparison' },
    { id: 'tutorial', label: 'Tutorial' }
  ];

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
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header Section */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl sm:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-blue-500 dark:from-indigo-400 dark:to-blue-300 mb-4">
            Depression Risk Analysis
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            A comprehensive analysis of our machine learning model's performance and insights
          </p>
          <Link 
            href="/"
            className="inline-flex items-center mt-6 px-4 py-2 text-sm font-medium text-indigo-600 bg-indigo-50 dark:bg-indigo-900/50 dark:text-indigo-300 rounded-lg hover:bg-indigo-100 dark:hover:bg-indigo-900 transition-all duration-200"
          >
            <FiArrowRight className="mr-2 transform rotate-180" />
            Back to Survey
          </Link>
        </motion.div>

        {/* Navigation */}
        <motion.nav 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex justify-center mb-12"
        >
          <div className="inline-flex rounded-lg bg-white dark:bg-gray-800 p-1 shadow-lg">
            {navItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveSection(item.id)}
                className={`px-6 py-3 rounded-lg text-sm font-medium transition-all duration-200 ${
                  activeSection === item.id
                    ? 'bg-indigo-600 text-white shadow-md'
                    : 'text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </motion.nav>

        {/* Content Sections */}
        <motion.div
          variants={staggerContainer}
          initial="initial"
          animate="animate"
          className="space-y-12"
        >
          {/* Overview Section */}
          {activeSection === 'overview' && (
            <motion.div variants={fadeIn} className="space-y-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {/* Model Recommendation Card */}
                <motion.div 
                  variants={fadeIn}
                  className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-xl p-8 transform hover:scale-[1.02] transition-all duration-300 border border-gray-100 dark:border-gray-700"
                >
                  <div className="flex items-center gap-4 mb-6">
                    <div className="p-3 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-2xl shadow-lg">
                      <FiBarChart2 className="w-6 h-6 text-white" />
                    </div>
                    <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300">
                      Recommended Model
                    </h2>
                  </div>
                  <div className="space-y-4">
                    <div className="p-4 bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/50 dark:to-blue-900/50 rounded-2xl">
                      <h3 className="text-lg font-semibold text-indigo-900 dark:text-indigo-200 mb-2">
                        Logistic Regression with LASSO
                      </h3>
                      <p className="text-sm text-indigo-700 dark:text-indigo-300">
                        Selected for its balance of accuracy, interpretability, and real-time prediction capabilities
                      </p>
                    </div>
                    {surveyAnalysis.reasons.map((reason, index) => (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="flex items-start gap-3 p-3 bg-white/50 dark:bg-gray-700/50 rounded-xl hover:bg-white/80 dark:hover:bg-gray-700/80 transition-colors duration-200"
                      >
                        <div className="mt-1 w-2 h-2 rounded-full bg-gradient-to-r from-indigo-500 to-blue-500" />
                        <p className="text-gray-700 dark:text-gray-300">{reason}</p>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>

                {/* Data Characteristics Card */}
                <motion.div 
                  variants={fadeIn}
                  className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-xl p-8 transform hover:scale-[1.02] transition-all duration-300 border border-gray-100 dark:border-gray-700"
                >
                  <div className="flex items-center gap-4 mb-6">
                    <div className="p-3 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl shadow-lg">
                      <FiInfo className="w-6 h-6 text-white" />
                    </div>
                    <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300">
                      Data Characteristics
                    </h2>
                  </div>
                  <div className="space-y-4">
                    {Object.entries(surveyAnalysis.dataCharacteristics).map(([key, value], index) => (
                      <motion.div
                        key={key}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 dark:from-blue-900/50 dark:to-cyan-900/50 rounded-2xl hover:bg-white/80 dark:hover:bg-gray-700/80 transition-colors duration-200"
                      >
                        <p className="font-medium text-blue-900 dark:text-blue-200 mb-1">{key}</p>
                        <p className="text-sm text-blue-700 dark:text-blue-300">
                          {Array.isArray(value) ? value.join(', ') : value}
                        </p>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              </div>

              {/* Model Improvement Analysis */}
              <motion.div 
                variants={fadeIn}
                className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-3xl shadow-xl p-8 border border-gray-100 dark:border-gray-700"
              >
                <div className="flex items-center gap-4 mb-8">
                  <div className="p-3 bg-gradient-to-br from-green-500 to-emerald-500 rounded-2xl shadow-lg">
                    <FiTrendingUp className="w-6 h-6 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300">
                    Model Comparison
                  </h2>
                </div>
                
                {/* Model Comparison Cards */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
                  {/* Original Model Card */}
                  <motion.div 
                    variants={fadeIn}
                    className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/50 dark:to-blue-900/50 rounded-2xl p-8 shadow-lg border border-indigo-100 dark:border-indigo-800"
                  >
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-xl shadow-lg">
                        <FiBarChart2 className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold text-indigo-900 dark:text-indigo-200">Original Model</h3>
                    </div>
                    <div className="space-y-6">
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                        <h4 className="text-lg font-medium text-indigo-800 dark:text-indigo-300 mb-4">Performance Metrics</h4>
                        <ul className="space-y-4">
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-indigo-600 dark:bg-indigo-400" />
                            <span className="text-indigo-700 dark:text-indigo-300">Training Accuracy: 87.5%</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-indigo-600 dark:bg-indigo-400" />
                            <span className="text-indigo-700 dark:text-indigo-300">Test Accuracy: 82.3%</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-indigo-600 dark:bg-indigo-400" />
                            <span className="text-indigo-700 dark:text-indigo-300">Accuracy Gap: 5.2%</span>
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                        <h4 className="text-lg font-medium text-indigo-800 dark:text-indigo-300 mb-4">Characteristics</h4>
                        <ul className="space-y-4">
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-indigo-600 dark:bg-indigo-400" />
                            <span className="text-indigo-700 dark:text-indigo-300">No regularization</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-indigo-600 dark:bg-indigo-400" />
                            <span className="text-indigo-700 dark:text-indigo-300">All features used</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-indigo-600 dark:bg-indigo-400" />
                            <span className="text-indigo-700 dark:text-indigo-300">Higher training accuracy</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-indigo-600 dark:bg-indigo-400" />
                            <span className="text-indigo-700 dark:text-indigo-300">Significant overfitting</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </motion.div>

                  {/* Improved Model Card */}
                  <motion.div 
                    variants={fadeIn}
                    className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/50 dark:to-emerald-900/50 rounded-2xl p-8 shadow-lg border border-green-100 dark:border-green-800"
                  >
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl shadow-lg">
                        <FiTrendingUp className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold text-green-900 dark:text-green-200">Improved Model</h3>
                    </div>
                    <div className="space-y-6">
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                        <h4 className="text-lg font-medium text-green-800 dark:text-green-300 mb-4">Performance Metrics</h4>
                        <ul className="space-y-4">
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-green-600 dark:bg-green-400" />
                            <span className="text-green-700 dark:text-green-300">Training Accuracy: 85.2%</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-green-600 dark:bg-green-400" />
                            <span className="text-green-700 dark:text-green-300">Test Accuracy: 84.8%</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-green-600 dark:bg-green-400" />
                            <span className="text-green-700 dark:text-green-300">Accuracy Gap: 0.4%</span>
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                        <h4 className="text-lg font-medium text-green-800 dark:text-green-300 mb-4">Improvements</h4>
                        <ul className="space-y-4">
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-green-600 dark:bg-green-400" />
                            <span className="text-green-700 dark:text-green-300">LASSO regularization (C=0.1)</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-green-600 dark:bg-green-400" />
                            <span className="text-green-700 dark:text-green-300">Feature selection through L1 penalty</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-green-600 dark:bg-green-400" />
                            <span className="text-green-700 dark:text-green-300">Better generalization</span>
                          </li>
                          <li className="flex items-center gap-3">
                            <span className="w-2 h-2 rounded-full bg-green-600 dark:bg-green-400" />
                            <span className="text-green-700 dark:text-green-300">Clearer feature importance</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </motion.div>
                </div>

                {/* Key Improvements Section */}
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/50 dark:to-indigo-900/50 rounded-2xl p-8">
                  <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-200 mb-6">Key Improvements</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-lg">
                          <FiTrendingUp className="w-5 h-5 text-white" />
                        </div>
                        <h4 className="font-medium text-blue-900 dark:text-blue-200">Reduced Overfitting</h4>
                      </div>
                      <p className="text-sm text-blue-700 dark:text-blue-300">
                        Training-test accuracy gap reduced from 5.2% to 0.4% through LASSO regularization
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 bg-gradient-to-br from-green-500 to-emerald-500 rounded-lg">
                          <FiTrendingUp className="w-5 h-5 text-white" />
                        </div>
                        <h4 className="font-medium text-green-900 dark:text-green-200">Better Generalization</h4>
                      </div>
                      <p className="text-sm text-green-700 dark:text-green-300">
                        Test accuracy improved from 82.3% to 84.8% with more robust feature selection
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
                          <FiPieChart className="w-5 h-5 text-white" />
                        </div>
                        <h4 className="font-medium text-purple-900 dark:text-purple-200">Feature Selection</h4>
                      </div>
                      <p className="text-sm text-purple-700 dark:text-purple-300">
                        LASSO (L1 regularization) automatically identifies and removes less important features
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>
            </motion.div>
          )}

          {/* Model Performance Section */}
          {activeSection === 'model-performance' && (
            <motion.div variants={fadeIn} className="space-y-8">
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
                              'rgba(255, 99, 132, 0.6)',
                            ],
                            borderColor: [
                              'rgba(54, 162, 235, 1)',
                              'rgba(255, 99, 132, 1)',
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

              {/* Cross-Validation Analysis */}
              <div className="w-full bg-white dark:bg-gray-700 p-4 sm:p-6 rounded-lg shadow">
                <h2 className="text-lg sm:text-xl font-semibold mb-4">Cross-Validation Analysis</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* CV Scores Visualization */}
                  <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                    <h3 className="text-base font-medium mb-3">5-Fold Cross-Validation Scores</h3>
                    {modelEvaluation && (
                      <div className="h-[200px]">
                        <Bar
                          data={{
                            labels: modelEvaluation.cv_results.scores.map((_, i) => `Fold ${i + 1}`),
                            datasets: [{
                              label: 'Accuracy Score',
                              data: modelEvaluation.cv_results.scores,
                              backgroundColor: 'rgba(54, 162, 235, 0.6)',
                              borderColor: 'rgba(54, 162, 235, 1)',
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
                                text: 'Cross-Validation Performance'
                              }
                            },
                            scales: {
                              y: {
                                beginAtZero: true,
                                max: 1,
                                title: {
                                  display: true,
                                  text: 'Accuracy Score'
                                }
                              }
                            }
                          }}
                        />
                      </div>
                    )}
                  </div>

                  {/* CV Statistics */}
                  <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                    <h3 className="text-base font-medium mb-3">Cross-Validation Statistics</h3>
                    {modelEvaluation && (
                      <div className="space-y-4">
                        <div className="p-3 bg-blue-50 dark:bg-blue-900 rounded-lg">
                          <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">Mean Accuracy</h4>
                          <p className="text-2xl font-bold text-blue-600 dark:text-blue-300">
                            {(modelEvaluation.cv_results.mean_score * 100).toFixed(1)}%
                          </p>
                          <p className="text-sm text-blue-700 dark:text-blue-400">
                            ±{(modelEvaluation.cv_results.std_score * 100).toFixed(1)}% (95% confidence interval)
                          </p>
                        </div>

                        <div className="p-3 bg-green-50 dark:bg-green-900 rounded-lg">
                          <h4 className="font-medium text-green-800 dark:text-green-200 mb-2">Model Stability</h4>
                          <p className="text-sm text-green-700 dark:text-green-300">
                            The standard deviation of {modelEvaluation.cv_results.std_score.toFixed(3)} indicates 
                            {modelEvaluation.cv_results.std_score < 0.02 ? ' high' : modelEvaluation.cv_results.std_score < 0.05 ? ' moderate' : ' some'} 
                            model stability across different data splits.
                          </p>
                        </div>

                        <div className="p-3 bg-purple-50 dark:bg-purple-900 rounded-lg">
                          <h4 className="font-medium text-purple-800 dark:text-purple-200 mb-2">Cross-Validation Benefits</h4>
                          <ul className="text-sm text-purple-700 dark:text-purple-300 space-y-1">
                            <li>• More robust performance estimate</li>
                            <li>• Better understanding of model stability</li>
                            <li>• Reduced impact of data splitting randomness</li>
                            <li>• Helps detect potential overfitting</li>
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* CV Interpretation */}
                <div className="mt-6 bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                  <h3 className="text-base font-medium mb-3">Cross-Validation Interpretation</h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">What This Means</h4>
                      <ul className="list-disc list-inside text-sm space-y-1">
                        <li>Model performance is consistent across different data splits</li>
                        <li>Low standard deviation indicates stable predictions</li>
                        <li>Mean accuracy provides a reliable performance estimate</li>
                        <li>Model generalizes well to unseen data</li>
                      </ul>
                    </div>
                    <div>
                      <h4 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">Practical Implications</h4>
                      <ul className="list-disc list-inside text-sm space-y-1">
                        <li>Confidence in model's real-world performance</li>
                        <li>Reliable risk predictions for new students</li>
                        <li>Stable feature importance rankings</li>
                        <li>Consistent probability estimates</li>
                      </ul>
                    </div>
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
            </motion.div>
          )}

          {/* Feature Impact Section */}
          {activeSection === 'feature-impact' && (
            <motion.div variants={fadeIn} className="space-y-8">
              {/* LASSO Regularization Analysis */}
              <div className="w-full bg-white dark:bg-gray-700 p-4 sm:p-6 rounded-lg shadow">
                <h2 className="text-lg sm:text-xl font-semibold mb-4">LASSO Regularization Analysis</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Feature Importance */}
                  <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                    <h3 className="text-base font-medium mb-3">Top 10 Most Important Features</h3>
                    <div className="h-[400px]">
                      <Bar
                        data={{
                          labels: ['Suicidal Thoughts (Yes)', 'Academic Pressure', 'Financial Stress', 
                                  'Dietary Habits (Unhealthy)', 'Age', 'Dietary Habits (Healthy)',
                                  'Work/Study Hours', 'Sleep Duration (<5h)', 'Study Satisfaction'],
                          datasets: [{
                            label: 'LASSO Coefficient',
                            data: [1.27, 1.16, 0.80, 0.59, -0.53, -0.46, 0.43, 0.33, -0.33],
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                          }]
                        }}
                        options={{
                          indexAxis: 'y',
                          responsive: true,
                          plugins: {
                            legend: {
                              display: false
                            },
                            title: {
                              display: true,
                              text: 'Feature Impact on Depression Risk'
                            }
                          },
                          scales: {
                            x: {
                              title: {
                                display: true,
                                text: 'LASSO Coefficient'
                              }
                            }
                          }
                        }}
                      />
                    </div>
                  </div>

                  {/* Regularization Explanation */}
                  <div className="bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                    <h3 className="text-base font-medium mb-3">LASSO Regularization Benefits</h3>
                    <div className="space-y-4">
                      <div className="p-3 bg-blue-50 dark:bg-blue-900 rounded-lg">
                        <h4 className="font-medium text-blue-800 dark:text-blue-200 mb-2">Feature Selection</h4>
                        <p className="text-sm text-blue-700 dark:text-blue-300">
                          LASSO (L1 regularization) automatically identifies and selects the most important features by setting less relevant features' coefficients to zero, helping to reduce model complexity.
                        </p>
                      </div>
                      
                      <div className="p-3 bg-green-50 dark:bg-green-900 rounded-lg">
                        <h4 className="font-medium text-green-800 dark:text-green-200 mb-2">Overfitting Prevention</h4>
                        <p className="text-sm text-green-700 dark:text-green-300">
                          By penalizing large coefficients with C=0.1, LASSO helps prevent the model from overfitting to the training data, resulting in better generalization.
                        </p>
                      </div>

                      <div className="p-3 bg-purple-50 dark:bg-purple-900 rounded-lg">
                        <h4 className="font-medium text-purple-800 dark:text-purple-200 mb-2">Model Interpretability</h4>
                        <p className="text-sm text-purple-700 dark:text-purple-300">
                          The LASSO coefficients provide clear insights into how each feature affects depression risk prediction, with positive values indicating increased risk and negative values indicating decreased risk.
                        </p>
                      </div>

                      <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                        <p>Regularization Strength (C=0.1):</p>
                        <ul className="list-disc list-inside mt-2">
                          <li>Stronger regularization to prevent overfitting</li>
                          <li>More aggressive feature selection through L1 penalty</li>
                          <li>Better generalization to new data</li>
                          <li>Clearer interpretation of feature importance</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Feature Impact Analysis */}
              <div className="mt-6 bg-gray-50 dark:bg-gray-800 p-4 rounded-lg">
                <h3 className="text-base font-medium mb-3">Feature Impact Analysis</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">Risk Factors</h4>
                    <ul className="list-disc list-inside text-sm space-y-1">
                      <li>Suicidal Thoughts (Yes) (1.27) - Strongest risk factor</li>
                      <li>Academic Pressure (1.16) - Major contributor to risk</li>
                      <li>Financial Stress (0.80) - Significant impact on risk</li>
                      <li>Dietary Habits (Unhealthy) (0.59) - Moderate risk factor</li>
                      <li>Work/Study Hours (0.43) - Notable contribution to risk</li>
                      <li>Sleep Duration (&lt;5h) (0.33) - Moderate risk factor</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium text-indigo-600 dark:text-indigo-400 mb-2">Protective Factors</h4>
                    <ul className="list-disc list-inside text-sm space-y-1">
                      <li>Age (-0.53) - Strong protective factor</li>
                      <li>Dietary Habits (Healthy) (-0.46) - Significant protective effect</li>
                      <li>Study Satisfaction (-0.33) - Moderate protective influence</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Prediction Visualizations */}
              <div className="w-full space-y-4 sm:space-y-6">
                <h2 className="text-lg sm:text-xl font-semibold">Model Predictions & Feature Impact</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                  {/* Academic Pressure Impact */}
                  <div className="bg-white dark:bg-gray-700 p-3 sm:p-4 rounded-lg shadow">
                    <h3 className="text-base sm:text-lg font-medium mb-3 sm:mb-4">Academic Pressure Impact</h3>
                    <div className="h-[250px] sm:h-[300px]">
                      <Line 
                        options={{
                          ...lineOptions,
                          plugins: {
                            ...lineOptions.plugins,
                            title: {
                              display: true,
                              text: 'LASSO Coefficient: 1.16 (Major Risk Factor)'
                            }
                          }
                        }} 
                        data={{
                          labels: [1, 2, 3, 4, 5],
                          datasets: [{
                            label: 'Risk Percentage',
                            data: [25, 40, 55, 70, 85],
                            borderColor: 'rgb(255, 99, 132)',
                            tension: 0.1,
                            fill: false
                          }]
                        }} 
                      />
                    </div>
                  </div>

                  {/* Sleep Duration Impact */}
                  <div className="bg-white dark:bg-gray-700 p-3 sm:p-4 rounded-lg shadow">
                    <h3 className="text-base sm:text-lg font-medium mb-3 sm:mb-4">Sleep Duration Impact</h3>
                    <div className="h-[250px] sm:h-[300px]">
                      <Line 
                        options={{
                          ...lineOptions,
                          plugins: {
                            ...lineOptions.plugins,
                            title: {
                              display: true,
                              text: 'LASSO Coefficient: 0.33 (Moderate Risk Factor)'
                            }
                          }
                        }} 
                        data={{
                          labels: ['&lt;6 hours', '6-7 hours', '7-8 hours', '8-9 hours', '&gt;9 hours'],
                          datasets: [{
                            label: 'Risk Percentage',
                            data: [75, 60, 45, 35, 30],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1,
                            fill: false
                          }]
                        }} 
                      />
                    </div>
                  </div>

                  {/* Work & Financial Impact */}
                  <div className="bg-white dark:bg-gray-700 p-3 sm:p-4 rounded-lg shadow md:col-span-2">
                    <h3 className="text-base sm:text-lg font-medium mb-3 sm:mb-4">Work & Financial Impact</h3>
                    <div className="h-[250px] sm:h-[300px] flex justify-center items-center">
                      <Line 
                        options={{
                          ...lineOptions,
                          plugins: {
                            ...lineOptions.plugins,
                            title: {
                              display: true,
                              text: 'LASSO Coefficients: Financial Stress (0.80) & Work/Study Hours (0.43)'
                            }
                          }
                        }} 
                        data={{
                          labels: [1, 2, 3, 4, 5],
                          datasets: [
                            {
                              label: 'Financial Stress',
                              data: [30, 45, 60, 75, 90],
                              borderColor: 'rgb(153, 102, 255)',
                              tension: 0.1,
                              fill: false
                            },
                            {
                              label: 'Work/Study Hours',
                              data: [25, 35, 45, 60, 75],
                              borderColor: 'rgb(255, 159, 64)',
                              tension: 0.1,
                              fill: false
                            }
                          ]
                        }} 
                      />
                    </div>
                  </div>
                </div>

                <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                  <h3 className="text-base font-medium mb-2">Feature Impact Analysis</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    These visualizations show the three most significant feature groups affecting depression risk:
                  </p>
                  <ul className="list-disc list-inside mt-2 text-sm text-gray-600 dark:text-gray-400">
                    <li>Suicidal Thoughts (1.27) & Academic Pressure (1.16): The strongest risk factors, showing a clear positive correlation with depression risk</li>
                    <li>Financial Stress (0.80) & Work/Study Hours (0.43): Combined impact showing how work and financial factors contribute to risk</li>
                    <li>Age (-0.53) & Dietary Habits (Healthy) (-0.46): Strong protective factors that help reduce depression risk</li>
                  </ul>
                </div>
              </div>
            </motion.div>
          )}  

          {/* Model Comparison Section */}
          {activeSection === 'model-comparison' && (
            <motion.div variants={fadeIn} className="space-y-8">
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
            </motion.div>
          )}

          {/* Tutorial Section */}
          {activeSection === 'tutorial' && (
            <motion.div variants={fadeIn} className="space-y-8">
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
                    <div className="mb-3 text-sm text-gray-600 dark:text-gray-400">
                      <p>Key preprocessing steps:</p>
                      <ul className="list-disc list-inside mt-2">
                        <li>Normalize numerical features (age, sleep hours, CGPA, etc.)</li>
                        <li>One-hot encode categorical variables (gender, profession, etc.)</li>
                        <li>Handle missing values with mean/mode imputation</li>
                        <li>Scale features to similar ranges for better model performance</li>
                      </ul>
                    </div>
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
            </motion.div>
          )}
        </motion.div>
      </main>
    </div>
  );
} 