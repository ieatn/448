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
import { FiArrowRight, FiBarChart2, FiPieChart, FiTrendingUp, FiInfo, FiUser } from 'react-icons/fi';

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
type ModelName = 'Logistic Regression' | 'KNN' | 'Neural Networks' | 'SVM';
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
    'Neural Networks': 0.85,
    'SVM': 0.81
  },
  training_time: {
    'Logistic Regression': 0.2,
    'KNN': 0.5,
    'Neural Networks': 1.2,
    'SVM': 0.7
  },
  interpretability: {
    'Logistic Regression': 0.95,
    'KNN': 0.70,
    'Neural Networks': 0.45,
    'SVM': 0.65
  },
  scalability: {
    'Logistic Regression': 0.90,
    'KNN': 0.60,
    'Neural Networks': 0.95,
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
  'Neural Networks': {
    'Accuracy': 0.85,
    'Speed': 0.75,
    'Interpretability': 0.45,
    'Feature Handling': 0.95,
    'Scalability': 0.95
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
    x: {
      beginAtZero: true,
      max: 1,
      ticks: {
        font: {
          size: 11
        }
      }
    },
    y: {
      beginAtZero: true,
      max: 1,
      ticks: {
        font: {
          size: 11
        }
      }
    }
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
      x: {
        beginAtZero: true,
        max: 1,
        ticks: {
          font: {
            size: 11
          }
        }
      },
      y: {
        beginAtZero: true,
        max: 1,
        ticks: {
          font: {
            size: 11
          }
        }
      }
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
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 pb-24 sm:pb-12">
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
          className="sm:flex sm:justify-center sm:mb-8 sm:mb-12"
        >
          {/* Mobile Navigation - Bottom Bar */}
          <div className="fixed bottom-0 left-0 right-0 bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl border-t border-gray-200 dark:border-gray-700 sm:hidden z-50">
            <div className="flex justify-around items-center h-16">
              {navItems.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setActiveSection(item.id)}
                  className={`flex flex-col items-center justify-center w-full h-full transition-all duration-200 ${
                    activeSection === item.id
                      ? 'text-indigo-600 dark:text-indigo-400'
                      : 'text-gray-600 dark:text-gray-400'
                  }`}
                >
                  <div className={`w-1 h-1 rounded-full mb-1 ${
                    activeSection === item.id
                      ? 'bg-indigo-600 dark:bg-indigo-400'
                      : 'bg-transparent'
                  }`} />
                  <span className="text-xs font-medium">{item.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Desktop Navigation */}
          <div className="hidden sm:inline-flex sm:rounded-lg bg-white dark:bg-gray-800 p-1 shadow-lg">
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
            <motion.div variants={fadeIn} className="space-y-6 sm:space-y-8">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8">
                {/* Model Recommendation Card */}
                <motion.div 
                  variants={fadeIn}
                  className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-xl p-6 sm:p-8 transform hover:scale-[1.02] transition-all duration-300 border border-gray-100 dark:border-gray-700"
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
                  className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-xl p-6 sm:p-8 transform hover:scale-[1.02] transition-all duration-300 border border-gray-100 dark:border-gray-700"
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

              {/* Key Features Box */}
              <motion.div 
                variants={fadeIn}
                className="bg-gradient-to-br from-blue-50/50 to-cyan-50/50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-2xl sm:rounded-3xl p-6 sm:p-8 backdrop-blur-sm"
              >
                <h3 className="text-lg sm:text-xl font-semibold text-blue-900 dark:text-blue-200 mb-4 sm:mb-6">Key Features</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
                  <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                    <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Probability-Based Predictions</h4>
                    <p className="text-sm text-blue-700 dark:text-blue-400">
                      Instead of just yes/no, we provide a percentage risk score, giving you a more nuanced understanding of your risk level.
                    </p>
                  </div>
                  <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                    <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Feature Importance</h4>
                    <p className="text-sm text-blue-700 dark:text-blue-400">
                      The model shows exactly how each factor (like sleep duration or academic pressure) affects your risk score.
                    </p>
                  </div>
                  <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                    <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Real-Time Analysis</h4>
                    <p className="text-sm text-blue-700 dark:text-blue-400">
                      Get instant results with our optimized model that processes your information in less than a second.
                    </p>
                  </div>
                </div>
              </motion.div>

              {/* Important Note Box */}
              <motion.div 
                variants={fadeIn}
                className="bg-gradient-to-br from-orange-50/50 to-red-50/50 dark:from-orange-900/20 dark:to-red-900/20 rounded-3xl p-8 backdrop-blur-sm"
              >
                <h3 className="text-xl font-semibold text-orange-900 dark:text-orange-200 mb-4">Important Note</h3>
                <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                  <p className="text-sm text-orange-700 dark:text-orange-400">
                    This tool is designed to provide insights and raise awareness about potential risk factors. It is not a substitute for professional medical advice. If you are concerned about your mental health, please consult a healthcare professional.
                  </p>
                </div>
              </motion.div>

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
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8 mb-8 sm:mb-12">
                  {/* Original Model Card */}
                  <motion.div 
                    variants={fadeIn}
                    className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/50 dark:to-blue-900/50 rounded-2xl p-6 sm:p-8 shadow-lg border border-indigo-100 dark:border-indigo-800"
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
                    className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/50 dark:to-emerald-900/50 rounded-2xl p-6 sm:p-8 shadow-lg border border-green-100 dark:border-green-800"
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
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/50 dark:to-indigo-900/50 rounded-2xl p-6 sm:p-8">
                  <h3 className="text-lg sm:text-xl font-semibold text-blue-900 dark:text-blue-200 mb-4 sm:mb-6">Key Improvements</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
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
            <motion.div variants={fadeIn} className="space-y-6 sm:space-y-8">
              {/* Performance Overview */}
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-xl p-6 sm:p-8 border border-gray-100 dark:border-gray-700">
                <div className="flex items-center gap-4 mb-8">
                  <div className="p-3 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-2xl shadow-lg">
                    <FiBarChart2 className="w-6 h-6 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300">
                    Model Performance
                  </h2>
                </div>

                {/* Performance Metrics Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 mb-6 sm:mb-8">
                  <div className="bg-gradient-to-br from-indigo-50 to-blue-50 dark:from-indigo-900/50 dark:to-blue-900/50 rounded-2xl p-6">
                    <h3 className="text-lg font-semibold text-indigo-900 dark:text-indigo-200 mb-2">Accuracy</h3>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold text-indigo-600 dark:text-indigo-300">84.8%</span>
                      <span className="text-sm text-indigo-500 dark:text-indigo-400">test accuracy</span>
                    </div>
                    <p className="mt-2 text-sm text-indigo-700 dark:text-indigo-300">
                      Consistent performance across training and test sets
                    </p>
                  </div>

                  <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/50 dark:to-emerald-900/50 rounded-2xl p-6">
                    <h3 className="text-lg font-semibold text-green-900 dark:text-green-200 mb-2">Precision</h3>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold text-green-600 dark:text-green-300">0.83</span>
                      <span className="text-sm text-green-500 dark:text-green-400">balanced</span>
                    </div>
                    <p className="mt-2 text-sm text-green-700 dark:text-green-300">
                      High precision in both depression and non-depression predictions
                    </p>
                  </div>

                  <div className="bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/50 dark:to-pink-900/50 rounded-2xl p-6">
                    <h3 className="text-lg font-semibold text-purple-900 dark:text-purple-200 mb-2">Recall</h3>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold text-purple-600 dark:text-purple-300">0.86</span>
                      <span className="text-sm text-purple-500 dark:text-purple-400">balanced</span>
                    </div>
                    <p className="mt-2 text-sm text-purple-700 dark:text-purple-300">
                      Strong ability to identify true depression cases
                    </p>
                  </div>
                </div>

                {/* LASSO Regularization Details */}
                <div className="bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/50 dark:to-indigo-900/50 rounded-2xl p-6 sm:p-8 mb-6 sm:mb-8">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-xl shadow-lg">
                      <FiTrendingUp className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-200">LASSO Regularization</h3>
                  </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                    <div className="space-y-4">
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-4">
                        <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Regularization Strength (C=0.1)</h4>
                        <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-400">
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Strong L1 penalty for feature selection
                          </li>
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Prevents overfitting effectively
                          </li>
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Promotes sparsity in coefficients
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-4">
                        <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Feature Selection Impact</h4>
                        <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-400">
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            15% of features automatically removed
                          </li>
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Clearer feature importance ranking
                          </li>
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Reduced model complexity
                          </li>
                        </ul>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-4">
                        <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Before LASSO</h4>
                        <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-400">
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Training Accuracy: 87.5%
                          </li>
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Test Accuracy: 82.3%
                          </li>
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Accuracy Gap: 5.2%
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-4">
                        <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">After LASSO</h4>
                        <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-400">
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Training Accuracy: 85.2%
                          </li>
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Test Accuracy: 84.8%
                          </li>
                          <li className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-blue-500" />
                            Accuracy Gap: 0.4%
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Cross-Validation Results */}
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/50 dark:to-emerald-900/50 rounded-2xl p-6 sm:p-8">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="p-3 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl shadow-lg">
                      <FiTrendingUp className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-xl font-semibold text-green-900 dark:text-green-200">Cross-Validation Results</h3>
                  </div>
                  
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                      <h4 className="font-medium text-green-800 dark:text-green-300 mb-4">5-Fold Cross-Validation</h4>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-green-700 dark:text-green-400">Mean Accuracy</span>
                          <span className="font-medium text-green-900 dark:text-green-200">84.5%</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-green-700 dark:text-green-400">Standard Deviation</span>
                          <span className="font-medium text-green-900 dark:text-green-200">±1.2%</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-green-700 dark:text-green-400">Fold Range</span>
                          <span className="font-medium text-green-900 dark:text-green-200">83.3% - 85.7%</span>
                        </div>
                      </div>
                    </div>
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                      <h4 className="font-medium text-green-800 dark:text-green-300 mb-4">Model Stability</h4>
                      <div className="space-y-4">
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 rounded-full bg-green-500" />
                          <span className="text-sm text-green-700 dark:text-green-400">Low variance across folds</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 rounded-full bg-green-500" />
                          <span className="text-sm text-green-700 dark:text-green-400">Consistent performance</span>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="w-2 h-2 rounded-full bg-green-500" />
                          <span className="text-sm text-green-700 dark:text-green-400">Robust to data splitting</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Feature Impact Section */}
          {activeSection === 'feature-impact' && (
            <motion.div variants={fadeIn} className="space-y-6 sm:space-y-8">
              {/* Main Feature Impact Card */}
              <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl p-6 sm:p-8 border border-gray-100/50 dark:border-gray-700/50">
                <div className="flex items-center gap-4 mb-8">
                  <div className="p-3 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-2xl shadow-lg">
                    <FiBarChart2 className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300">
                      Feature Impact Analysis
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">
                      Understanding how different factors influence depression risk
                    </p>
                  </div>
                </div>

                {/* Feature Importance Chart - Full Width */}
                <div className="bg-gradient-to-br from-indigo-50/50 to-blue-50/50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-2xl sm:rounded-3xl p-4 sm:p-8 backdrop-blur-sm mb-6 sm:mb-8">
                  <h3 className="text-lg sm:text-xl font-semibold text-indigo-900 dark:text-indigo-200 mb-4 sm:mb-6 text-center">Feature Impact on Depression Risk</h3>
                  <div className="w-full h-[400px] sm:h-[500px]">
                    <Bar
                      data={{
                        labels: ['Suicidal Thoughts (Yes)', 'Academic Pressure', 'Financial Stress', 
                                'Dietary Habits (Unhealthy)', 'Age', 'Dietary Habits (Healthy)',
                                'Work/Study Hours', 'Sleep Duration (<5h)', 'Study Satisfaction'],
                        datasets: [{
                          label: 'LASSO Coefficient',
                          data: [1.27, 1.16, 0.80, 0.59, -0.53, -0.46, 0.43, 0.33, -0.33],
                          backgroundColor: [
                            'rgba(239, 68, 68, 0.7)',
                            'rgba(239, 68, 68, 0.7)',
                            'rgba(239, 68, 68, 0.7)',
                            'rgba(239, 68, 68, 0.7)',
                            'rgba(34, 197, 94, 0.7)',
                            'rgba(34, 197, 94, 0.7)',
                            'rgba(239, 68, 68, 0.7)',
                            'rgba(239, 68, 68, 0.7)',
                            'rgba(34, 197, 94, 0.7)',
                          ],
                          borderColor: [
                            'rgb(239, 68, 68)',
                            'rgb(239, 68, 68)',
                            'rgb(239, 68, 68)',
                            'rgb(239, 68, 68)',
                            'rgb(34, 197, 94)',
                            'rgb(34, 197, 94)',
                            'rgb(239, 68, 68)',
                            'rgb(239, 68, 68)',
                            'rgb(34, 197, 94)',
                          ],
                          borderWidth: 2,
                          borderRadius: 4
                        }]
                      }}
                      options={{
                        indexAxis: 'y',
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                          legend: {
                            display: false
                          },
                          title: {
                            display: false
                          }
                        },
                        scales: {
                          x: {
                            grid: {
                              display: false
                            },
                            border: {
                              display: false
                            },
                            ticks: {
                              font: {
                                size: 12
                              }
                            }
                          },
                          y: {
                            grid: {
                              display: false
                            },
                            border: {
                              display: false
                            },
                            ticks: {
                              font: {
                                size: 12
                              },
                              padding: 10,
                              callback: function(value, index) {
                                const label = this.getLabelForValue(index);
                                return label.length > 25 ? label.substring(0, 25) + '...' : label;
                              }
                            }
                          }
                        },
                        layout: {
                          padding: {
                            left: 10,
                            right: 10
                          }
                        }
                      }}
                    />
                  </div>
                </div>

                {/* Risk and Protective Factors Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8">
                  {/* Risk Factors */}
                  <div className="bg-gradient-to-br from-red-50/50 to-pink-50/50 dark:from-red-900/20 dark:to-pink-900/20 rounded-3xl p-8 backdrop-blur-sm">
                    <h3 className="text-xl font-semibold text-red-900 dark:text-red-200 mb-6">Risk Factors</h3>
                    <div className="space-y-4">
                      {featureImpactAnalysis.positiveImpact.map((factor, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className="flex items-start gap-4 p-4 bg-white/50 dark:bg-gray-800/50 rounded-2xl hover:bg-white/80 dark:hover:bg-gray-800/80 transition-all duration-200"
                        >
                          <div className="mt-1 w-2 h-2 rounded-full bg-gradient-to-r from-red-500 to-pink-500" />
                          <div>
                            <p className="font-semibold text-red-900 dark:text-red-200">{factor.feature}</p>
                            <p className="text-sm text-red-700 dark:text-red-300 mt-1">{factor.description}</p>
                            <p className="text-xs font-medium text-red-600 dark:text-red-400 mt-2">Impact: {factor.coefficient}</p>
                          </div>
                        </motion.div>
                      ))}
                  </div>
                </div>

                  {/* Protective Factors */}
                  <div className="bg-gradient-to-br from-green-50/50 to-emerald-50/50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-3xl p-8 backdrop-blur-sm">
                    <h3 className="text-xl font-semibold text-green-900 dark:text-green-200 mb-6">Protective Factors</h3>
                    <div className="space-y-4">
                      {featureImpactAnalysis.negativeImpact.map((factor, index) => (
                        <motion.div
                          key={index}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className="flex items-start gap-4 p-4 bg-white/50 dark:bg-gray-800/50 rounded-2xl hover:bg-white/80 dark:hover:bg-gray-800/80 transition-all duration-200"
                        >
                          <div className="mt-1 w-2 h-2 rounded-full bg-gradient-to-r from-green-500 to-emerald-500" />
                          <div>
                            <p className="font-semibold text-green-900 dark:text-green-200">{factor.feature}</p>
                            <p className="text-sm text-green-700 dark:text-green-300 mt-1">{factor.description}</p>
                            <p className="text-xs font-medium text-green-600 dark:text-green-400 mt-2">Impact: {factor.coefficient}</p>
                          </div>
                        </motion.div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Key Insights */}
                <div className="mt-6 sm:mt-8 bg-gradient-to-br from-gray-50/50 to-gray-100/50 dark:from-gray-800/20 dark:to-gray-700/20 rounded-2xl sm:rounded-3xl p-6 sm:p-8 backdrop-blur-sm">
                  <h3 className="text-lg sm:text-xl font-semibold text-gray-900 dark:text-gray-200 mb-4 sm:mb-6">Key Insights</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-6 hover:bg-white/80 dark:hover:bg-gray-800/80 transition-all duration-200">
                      <div className="flex items-center gap-4 mb-4">
                        <div className="p-3 bg-gradient-to-br from-red-500 to-pink-500 rounded-xl shadow-lg">
                          <FiTrendingUp className="w-6 h-6 text-white" />
                        </div>
                        <h4 className="text-lg font-semibold text-red-900 dark:text-red-200">Strongest Risk Factors</h4>
                      </div>
                      <p className="text-sm text-red-700 dark:text-red-300 leading-relaxed">
                        Suicidal thoughts and academic pressure show the strongest positive correlation with depression risk.
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-6 hover:bg-white/80 dark:hover:bg-gray-800/80 transition-all duration-200">
                      <div className="flex items-center gap-4 mb-4">
                        <div className="p-3 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl shadow-lg">
                          <FiTrendingUp className="w-6 h-6 text-white" />
                        </div>
                        <h4 className="text-lg font-semibold text-green-900 dark:text-green-200">Protective Factors</h4>
                      </div>
                      <p className="text-sm text-green-700 dark:text-green-300 leading-relaxed">
                        Age and healthy dietary habits show strong protective effects against depression risk.
                      </p>
                    </div>
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-6 hover:bg-white/80 dark:hover:bg-gray-800/80 transition-all duration-200">
                      <div className="flex items-center gap-4 mb-4">
                        <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-xl shadow-lg">
                          <FiPieChart className="w-6 h-6 text-white" />
                        </div>
                        <h4 className="text-lg font-semibold text-blue-900 dark:text-blue-200">Moderate Factors</h4>
                      </div>
                      <p className="text-sm text-blue-700 dark:text-blue-300 leading-relaxed">
                        Sleep duration and work/study hours show moderate but significant impact on risk levels.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Model Comparison Section */}
          {activeSection === 'model-comparison' && (
            <motion.div variants={fadeIn} className="space-y-6 sm:space-y-8">
              {/* Main Model Comparison Card */}
              <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl p-6 sm:p-8 border border-gray-100/50 dark:border-gray-700/50">
                <div className="flex items-center gap-4 mb-8">
                  <div className="p-3 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-2xl shadow-lg">
                    <FiBarChart2 className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300">
                      Model Comparison
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">
                      Interactive comparison of different machine learning models
                    </p>
                  </div>
                </div>

                {/* Interactive Model Comparison Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8 mb-8 sm:mb-12">
                  {/* Performance Metrics Chart */}
                  <div className="bg-gradient-to-br from-indigo-50/50 to-blue-50/50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-3xl p-4 sm:p-8 backdrop-blur-sm">
                    <h3 className="text-xl font-semibold text-indigo-900 dark:text-indigo-200 mb-6">Performance Metrics</h3>
                    <div className="h-[300px] sm:h-[400px] w-full">
                      <Bar
                        data={performanceData}
                        options={{
                          ...barOptions,
                          maintainAspectRatio: false,
                          responsive: true,
                          plugins: {
                            ...barOptions.plugins,
                            legend: {
                              position: 'top' as const,
                              labels: {
                                boxWidth: 12,
                                padding: 15,
                                font: {
                                  size: 12
                                }
                              }
                            }
                          },
                          scales: {
                            x: {
                              ...barOptions.scales.x,
                              ticks: {
                                font: {
                                  size: 11
                                },
                                maxRotation: 45,
                                minRotation: 45
                              }
                            },
                            y: {
                              ...barOptions.scales.y,
                              ticks: {
                                font: {
                                  size: 11
                                }
                              }
                            }
                          }
                        }}
                      />
                    </div>
                    <div className="mt-4 flex flex-wrap justify-center gap-2">
                      {Object.keys(modelPerformance).map((metric) => (
                        <button
                          key={metric}
                          onClick={() => setSelectedMetric(metric as MetricName)}
                          className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all duration-200 ${
                            selectedMetric === metric
                              ? 'bg-indigo-600 text-white shadow-md'
                              : 'bg-white/50 text-gray-600 dark:text-gray-300 hover:bg-white/80'
                          }`}
                        >
                          {metric.charAt(0).toUpperCase() + metric.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>
                  
                  {/* Model Characteristics Radar Chart */}
                  <div className="bg-gradient-to-br from-green-50/50 to-emerald-50/50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-3xl p-4 sm:p-8 backdrop-blur-sm">
                    <h3 className="text-xl font-semibold text-green-900 dark:text-green-200 mb-6">Model Characteristics</h3>
                    <div className="h-[300px] sm:h-[400px] w-full">
                      <Radar
                        data={radarData}
                        options={{
                          ...radarOptions,
                          maintainAspectRatio: false,
                          responsive: true,
                          plugins: {
                            legend: {
                              position: 'top' as const,
                              labels: {
                                boxWidth: 12,
                                padding: 15,
                                font: {
                                  size: 12
                                }
                              }
                            }
                          },
                          scales: {
                            r: {
                              ...radarOptions.scales.r,
                              ticks: {
                                font: {
                                  size: 11
                                },
                                backdropColor: 'transparent'
                              },
                              pointLabels: {
                                font: {
                                  size: 11
                                }
                              }
                            }
                          }
                        }}
                      />
                    </div>
                    <div className="mt-4 flex flex-wrap justify-center gap-2">
                      {Object.keys(modelCharacteristics).map((model) => (
                        <button
                          key={model}
                          onClick={() => setSelectedModel(model as ModelName)}
                          className={`px-3 py-1.5 rounded-lg text-xs sm:text-sm font-medium transition-all duration-200 ${
                            selectedModel === model
                              ? 'bg-green-600 text-white shadow-md'
                              : 'bg-white/50 text-gray-600 dark:text-gray-300 hover:bg-white/80'
                          }`}
                        >
                          {model}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Model Analysis Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8">
                  {/* KNN */}
                  <div className="bg-gradient-to-br from-blue-50/50 to-indigo-50/50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-3xl p-8 backdrop-blur-sm">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-gradient-to-br from-blue-500 to-indigo-500 rounded-xl shadow-lg">
                        <FiPieChart className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-200">K-Nearest Neighbors</h3>
                    </div>
                    <div className="space-y-4">
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-4">
                        <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Advantages</h4>
                        <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                            <span>Simple to understand and implement</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                            <span>Can capture complex, non-linear relationships</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                            <span>No training required, just stores training data</span>
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-4">
                        <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-2">Limitations</h4>
                        <ul className="space-y-2 text-sm text-blue-700 dark:text-blue-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                            <span>Computationally expensive with large datasets</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                            <span>Less interpretable predictions</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                            <span>Sensitive to feature scaling</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  {/* SVM */}
                  <div className="bg-gradient-to-br from-purple-50/50 to-pink-50/50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-3xl p-8 backdrop-blur-sm">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl shadow-lg">
                        <FiPieChart className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold text-purple-900 dark:text-purple-200">Support Vector Machines</h3>
                    </div>
                    <div className="space-y-4">
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-4">
                        <h4 className="font-medium text-purple-800 dark:text-purple-300 mb-2">Advantages</h4>
                        <ul className="space-y-2 text-sm text-purple-700 dark:text-purple-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                            <span>Effective in high-dimensional spaces</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                            <span>Powerful for classification tasks</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                            <span>Robust against overfitting</span>
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-4">
                        <h4 className="font-medium text-purple-800 dark:text-purple-300 mb-2">Limitations</h4>
                        <ul className="space-y-2 text-sm text-purple-700 dark:text-purple-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                            <span>Less interpretable than Logistic Regression</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                            <span>Can be slow for very large datasets</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                            <span>Sensitive to parameter tuning</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  {/* Neural Networks */}
                  <div className="bg-gradient-to-br from-red-50/50 to-orange-50/50 dark:from-red-900/20 dark:to-orange-900/20 rounded-3xl p-8 backdrop-blur-sm">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-gradient-to-br from-red-500 to-orange-500 rounded-xl shadow-lg">
                        <FiPieChart className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold text-red-900 dark:text-red-200">Neural Networks</h3>
                    </div>
                    <div className="space-y-4">
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-4">
                        <h4 className="font-medium text-red-800 dark:text-red-300 mb-2">Advantages</h4>
                        <ul className="space-y-2 text-sm text-red-700 dark:text-red-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" />
                            <span>Can learn extremely complex patterns</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" />
                            <span>Potential for very high accuracy</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" />
                            <span>Excellent for large datasets</span>
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-4">
                        <h4 className="font-medium text-red-800 dark:text-red-300 mb-2">Limitations</h4>
                        <ul className="space-y-2 text-sm text-red-700 dark:text-red-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" />
                            <span>Requires large amounts of data</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" />
                            <span>Computationally intensive to train</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0" />
                            <span>Low interpretability ("black box")</span>
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Why Logistic Regression Section */}
                <div className="mt-6 sm:mt-8 bg-gradient-to-br from-green-50/50 to-emerald-50/50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-2xl sm:rounded-3xl p-6 sm:p-8 backdrop-blur-sm">
                  <div className="flex items-center gap-4 mb-6">
                    <div className="p-3 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl shadow-lg">
                      <FiTrendingUp className="w-6 h-6 text-white" />
                    </div>
                    <h3 className="text-xl font-semibold text-green-900 dark:text-green-200">Why Logistic Regression?</h3>
                  </div>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-6">
                      <h4 className="font-medium text-green-800 dark:text-green-300 mb-4">Key Benefits</h4>
                      <ul className="space-y-3 text-sm text-green-700 dark:text-green-400">
                        <li className="flex items-start gap-3">
                          <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                          <span>Provides clear, interpretable probability-based predictions</span>
                        </li>
                        <li className="flex items-start gap-3">
                          <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                          <span>LASSO regularization helps identify key risk factors</span>
                        </li>
                        <li className="flex items-start gap-3">
                          <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                          <span>Fast computation time for real-time predictions</span>
                        </li>
                        <li className="flex items-start gap-3">
                          <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                          <span>Clear interpretation of feature importance through coefficients</span>
                        </li>
                      </ul>
                    </div>
                    <div className="bg-white/50 dark:bg-gray-800/50 rounded-2xl p-6">
                      <h4 className="font-medium text-green-800 dark:text-green-300 mb-4">Performance Metrics</h4>
                      <div className="space-y-4">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-green-700 dark:text-green-400">Accuracy</span>
                          <span className="font-medium text-green-900 dark:text-green-200">84.8%</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-green-700 dark:text-green-400">Training Time</span>
                          <span className="font-medium text-green-900 dark:text-green-200">0.2s</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-green-700 dark:text-green-400">Interpretability</span>
                          <span className="font-medium text-green-900 dark:text-green-200">95%</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-green-700 dark:text-green-400">Scalability</span>
                          <span className="font-medium text-green-900 dark:text-green-200">90%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Tutorial Section */}
          {activeSection === 'tutorial' && (
            <motion.div variants={fadeIn} className="space-y-6 sm:space-y-8">
              <div className="bg-white/90 dark:bg-gray-800/90 backdrop-blur-xl rounded-2xl sm:rounded-3xl shadow-2xl p-6 sm:p-8 border border-gray-100/50 dark:border-gray-700/50">
                <div className="flex items-center gap-4 mb-8">
                  <div className="p-3 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-2xl shadow-lg">
                    <FiBarChart2 className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h2 className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-300">
                      How Our Model Works
                    </h2>
                    <p className="text-gray-600 dark:text-gray-400 mt-1">
                      A step-by-step guide to our depression risk prediction system
                    </p>
                  </div>
                </div>

                <div className="space-y-6 sm:space-y-8">
                  {/* Step 1: Data Collection */}
                  <div className="bg-gradient-to-br from-indigo-50/50 to-blue-50/50 dark:from-indigo-900/20 dark:to-blue-900/20 rounded-2xl sm:rounded-3xl p-6 sm:p-8">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-gradient-to-br from-indigo-500 to-blue-500 rounded-xl shadow-lg">
                        <FiUser className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold text-indigo-900 dark:text-indigo-200">1. Data Collection</h3>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8">
                      <div className="space-y-4">
                        <p className="text-gray-700 dark:text-gray-300">
                          We collect information about various factors that can influence mental health:
                        </p>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-500 flex-shrink-0" />
                            <span>Personal information (age, gender)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-500 flex-shrink-0" />
                            <span>Academic factors (CGPA, study satisfaction)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-500 flex-shrink-0" />
                            <span>Lifestyle factors (sleep duration, dietary habits)</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-indigo-500 flex-shrink-0" />
                            <span>Mental health history (suicidal thoughts, family history)</span>
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                        <h4 className="font-medium text-indigo-800 dark:text-indigo-300 mb-4">Implementation</h4>
                        <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-xl overflow-x-auto text-sm">
{`# Data handling and web framework
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load the dataset
data = pd.read_csv('student_depression_dataset.csv')
print("Successfully loaded 'student_depression_dataset.csv'")`}
                        </pre>
                      </div>
                    </div>
                  </div>

                  {/* Step 2: Data Processing */}
                  <div className="bg-gradient-to-br from-green-50/50 to-emerald-50/50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-2xl sm:rounded-3xl p-6 sm:p-8">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl shadow-lg">
                        <FiTrendingUp className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold text-green-900 dark:text-green-200">2. Data Processing</h3>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8">
                      <div className="space-y-4">
                        <p className="text-gray-700 dark:text-gray-300">
                          Before analysis, we prepare the data to ensure accurate predictions:
                        </p>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                            <span>Convert all measurements to a common scale</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                            <span>Handle any missing information</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-green-500 flex-shrink-0" />
                            <span>Organize categorical data (like gender, profession)</span>
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                        <h4 className="font-medium text-green-800 dark:text-green-300 mb-4">Implementation</h4>
                        <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-xl overflow-x-auto text-sm">
{`# Define feature types
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
    ])`}
                        </pre>
                      </div>
                    </div>
                  </div>

                  {/* Step 3: Model Training */}
                  <div className="bg-gradient-to-br from-purple-50/50 to-pink-50/50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-2xl sm:rounded-3xl p-6 sm:p-8">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl shadow-lg">
                        <FiPieChart className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold text-purple-900 dark:text-purple-200">3. Model Training</h3>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8">
                      <div className="space-y-4">
                        <p className="text-gray-700 dark:text-gray-300">
                          Our model learns from over 27,000 student records to identify patterns:
                        </p>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                            <span>Uses LASSO regularization to focus on important factors</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                            <span>Automatically identifies key risk factors</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                            <span>Maintains 84.8% accuracy on test data</span>
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                        <h4 className="font-medium text-purple-800 dark:text-purple-300 mb-4">Implementation</h4>
                        <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-xl overflow-x-auto text-sm">
{`# Create and train the model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(
        random_state=42,
        solver='liblinear',
        penalty='l1',  # LASSO regularization
        C=0.1,  # Strong regularization
        max_iter=1000
    ))
])

# Train the model
model_pipeline.fit(X_train, y_train)

# Evaluate performance
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")`}
                        </pre>
                      </div>
                    </div>
                  </div>

                  {/* Step 4: Making Predictions */}
                  <div className="bg-gradient-to-br from-blue-50/50 to-cyan-50/50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-2xl sm:rounded-3xl p-6 sm:p-8">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-xl shadow-lg">
                        <FiTrendingUp className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold text-blue-900 dark:text-blue-200">4. Making Predictions</h3>
                    </div>
                    
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 sm:gap-8">
                      <div className="space-y-4">
                        <p className="text-gray-700 dark:text-gray-300">
                          When you submit your information, our model:
                        </p>
                        <ul className="space-y-2 text-gray-600 dark:text-gray-400">
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                            <span>Processes your input through the same pipeline</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                            <span>Calculates your risk probability</span>
                          </li>
                          <li className="flex items-start gap-2">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                            <span>Provides detailed insights about risk factors</span>
                          </li>
                        </ul>
                      </div>
                      <div className="bg-white/50 dark:bg-gray-800/50 rounded-xl p-6">
                        <h4 className="font-medium text-blue-800 dark:text-blue-300 mb-4">Implementation</h4>
                        <pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-xl overflow-x-auto text-sm">
{`@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json(force=True)
    
    # Convert to DataFrame
    input_df = pd.DataFrame([data])
    
    # Get prediction probabilities
    proba = model_pipeline.predict_proba(input_df)
    depression_prob = proba[0][1] * 100
    
    # Get binary prediction
    prediction = model_pipeline.predict(input_df)[0]
    
    return jsonify({
        "predicted_depression": bool(prediction),
        "probability_of_depression": round(depression_prob, 2)
    })`}
                        </pre>
                      </div>
                    </div>
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