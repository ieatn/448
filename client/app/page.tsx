'use client';

import { useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { FiArrowRight, FiUser, FiBook, FiClock, FiHeart, FiDollarSign, FiHome } from 'react-icons/fi';

interface PredictionResult {
  percentage: number;
  message: string;
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

export default function Home() {
  const [age, setAge] = useState('');
  const [gender, setGender] = useState('');
  const [profession, setProfession] = useState('');
  const [academicPressure, setAcademicPressure] = useState('');
  const [workPressure, setWorkPressure] = useState('');
  const [cgpa, setCgpa] = useState('');
  const [studySatisfaction, setStudySatisfaction] = useState('');
  const [jobSatisfaction, setJobSatisfaction] = useState('');
  const [sleepDuration, setSleepDuration] = useState('');
  const [dietaryHabits, setDietaryHabits] = useState('');
  const [degree, setDegree] = useState('');
  const [suicidalThoughts, setSuicidalThoughts] = useState('');
  const [workStudyHours, setWorkStudyHours] = useState('');
  const [financialStress, setFinancialStress] = useState('');
  const [familyHistory, setFamilyHistory] = useState('');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const testServer = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });
      const data = await response.text();
      console.log('Server response:', data);
    } catch (err) {
      console.error('Failed to connect to server:', err);
      alert('Failed to connect to server. Check console for details.');
    }
  };

  const fillSampleData = () => {
    setAge('25');
    setGender('Female');
    setProfession('Student');
    setAcademicPressure('3');
    setWorkPressure('1');
    setCgpa('7.5');
    setStudySatisfaction('4');
    setJobSatisfaction('1');
    setSleepDuration('7-8 hours');
    setDietaryHabits('Healthy');
    setDegree('BSc');
    setSuicidalThoughts('No');
    setWorkStudyHours('6');
    setFinancialStress('2');
    setFamilyHistory('No');
  };

  const fillDepressionSampleData = () => {
    setAge('22');
    setGender('Male');
    setProfession('Student');
    setAcademicPressure('5');
    setWorkPressure('4');
    setCgpa('6.2');
    setStudySatisfaction('2');
    setJobSatisfaction('2');
    setSleepDuration('<6 hours');
    setDietaryHabits('Unhealthy');
    setDegree('BSc');
    setSuicidalThoughts('Yes');
    setWorkStudyHours('12');
    setFinancialStress('5');
    setFamilyHistory('Yes');
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    const formData = {
      Age: parseInt(age),
      Gender: gender,
      Profession: profession,
      'Academic Pressure': parseFloat(academicPressure),
      'Work Pressure': parseFloat(workPressure),
      CGPA: parseFloat(cgpa),
      'Study Satisfaction': parseFloat(studySatisfaction),
      'Job Satisfaction': parseFloat(jobSatisfaction),
      'Sleep Duration': sleepDuration,
      'Dietary Habits': dietaryHabits,
      Degree: degree,
      'Have you ever had suicidal thoughts ?': suicidalThoughts,
      'Work/Study Hours': parseFloat(workStudyHours),
      'Financial Stress': parseFloat(financialStress),
      'Family History of Mental Illness': familyHistory
    };

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Network response was not ok');
      }

      const result = await response.json();
      
      console.log('Prediction result:', {
        predicted_depression: result.predicted_depression,
        probability: result.probability_of_depression,
        raw_result: result
      });
      
      setPrediction({
        percentage: result.probability_of_depression,
        message: result.predicted_depression 
          ? "High Risk - You may be at increased risk for depression. Consider speaking with a healthcare professional."
          : "Low Risk - Few risk factors present. Continue maintaining healthy habits."
      });

      // Add a small delay to ensure the prediction box is rendered
      setTimeout(() => {
        const predictionBox = document.getElementById('prediction-box');
        if (predictionBox) {
          predictionBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 100);

    } catch (err) {
      console.error('Failed to fetch prediction:', err);
      setError(err instanceof Error ? err.message : 'Failed to get prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header Section */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-4xl sm:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-blue-500 dark:from-indigo-400 dark:to-blue-300 mb-4">
            Depression Risk Assessment
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Complete this survey to assess your risk level and receive personalized insights
          </p>
        </motion.div>

        {/* Sample Data Buttons */}
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex justify-center gap-4 mb-8"
        >
          <button
            onClick={fillSampleData}
            className="px-4 py-2 text-sm font-medium text-indigo-600 bg-indigo-50 dark:bg-indigo-900/50 dark:text-indigo-300 rounded-lg hover:bg-indigo-100 dark:hover:bg-indigo-900 transition-all duration-200"
          >
            Fill Low Risk Sample
          </button>
          <button
            onClick={fillDepressionSampleData}
            className="px-4 py-2 text-sm font-medium text-red-600 bg-red-50 dark:bg-red-900/50 dark:text-red-300 rounded-lg hover:bg-red-100 dark:hover:bg-red-900 transition-all duration-200"
          >
            Fill High Risk Sample
          </button>
        </motion.div>

        {/* Form Section */}
        <motion.form
          variants={staggerContainer}
          initial="initial"
          animate="animate"
          onSubmit={handleSubmit}
          className="space-y-8"
        >
          {/* Personal Information Section */}
          <motion.div variants={fadeIn} className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-4 mb-6">
              <div className="p-3 bg-indigo-100 dark:bg-indigo-900 rounded-xl">
                <FiUser className="w-6 h-6 text-indigo-600 dark:text-indigo-400" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Personal Information</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="age" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Age
                </label>
                <input
                  type="number"
                  id="age"
                  name="age"
                  value={age}
                  onChange={(e) => setAge(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                />
              </div>
              <div>
                <label htmlFor="gender" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Gender
                </label>
                <select
                  id="gender"
                  name="gender"
                  value={gender}
                  onChange={(e) => setGender(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select gender</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </div>
              <div>
                <label htmlFor="profession" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Profession
                </label>
                <select
                  id="profession"
                  name="profession"
                  value={profession}
                  onChange={(e) => setProfession(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select profession</option>
                  <option value="Student">Student</option>
                  <option value="Working Professional">Working Professional</option>
                  <option value="Self-employed">Self-employed</option>
                  <option value="Unemployed">Unemployed</option>
                </select>
              </div>
              <div>
                <label htmlFor="degree" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Degree
                </label>
                <select
                  id="degree"
                  name="degree"
                  value={degree}
                  onChange={(e) => setDegree(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select degree</option>
                  <option value="BSc">BSc</option>
                  <option value="MSc">MSc</option>
                  <option value="PhD">PhD</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>
          </motion.div>

          {/* Academic Information Section */}
          <motion.div variants={fadeIn} className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-4 mb-6">
              <div className="p-3 bg-blue-100 dark:bg-blue-900 rounded-xl">
                <FiBook className="w-6 h-6 text-blue-600 dark:text-blue-400" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Academic Information</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="cgpa" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  CGPA
                </label>
                <input
                  type="number"
                  id="cgpa"
                  name="cgpa"
                  value={cgpa}
                  onChange={(e) => setCgpa(e.target.value)}
                  step="0.01"
                  min="0"
                  max="10"
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                />
              </div>
              <div>
                <label htmlFor="academic_pressure" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Academic Pressure
                </label>
                <select
                  id="academic_pressure"
                  name="academic_pressure"
                  value={academicPressure}
                  onChange={(e) => setAcademicPressure(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select pressure level</option>
                  <option value="1">Very Low</option>
                  <option value="2">Low</option>
                  <option value="3">Moderate</option>
                  <option value="4">High</option>
                  <option value="5">Very High</option>
                </select>
              </div>
              <div>
                <label htmlFor="study_satisfaction" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Study Satisfaction
                </label>
                <select
                  id="study_satisfaction"
                  name="study_satisfaction"
                  value={studySatisfaction}
                  onChange={(e) => setStudySatisfaction(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select satisfaction level</option>
                  <option value="1">Very Dissatisfied</option>
                  <option value="2">Dissatisfied</option>
                  <option value="3">Neutral</option>
                  <option value="4">Satisfied</option>
                  <option value="5">Very Satisfied</option>
                </select>
              </div>
              <div>
                <label htmlFor="work_study_hours" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Work/Study Hours per Day
                </label>
                <input
                  type="number"
                  id="work_study_hours"
                  name="work_study_hours"
                  value={workStudyHours}
                  onChange={(e) => setWorkStudyHours(e.target.value)}
                  min="0"
                  max="24"
                  step="0.5"
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                />
              </div>
            </div>
          </motion.div>

          {/* Work & Financial Information Section */}
          <motion.div variants={fadeIn} className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-4 mb-6">
              <div className="p-3 bg-green-100 dark:bg-green-900 rounded-xl">
                <FiDollarSign className="w-6 h-6 text-green-600 dark:text-green-400" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Work & Financial Information</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="work_pressure" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Work Pressure
                </label>
                <select
                  id="work_pressure"
                  name="work_pressure"
                  value={workPressure}
                  onChange={(e) => setWorkPressure(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select pressure level</option>
                  <option value="1">Very Low</option>
                  <option value="2">Low</option>
                  <option value="3">Moderate</option>
                  <option value="4">High</option>
                  <option value="5">Very High</option>
                </select>
              </div>
              <div>
                <label htmlFor="job_satisfaction" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Job Satisfaction
                </label>
                <select
                  id="job_satisfaction"
                  name="job_satisfaction"
                  value={jobSatisfaction}
                  onChange={(e) => setJobSatisfaction(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select satisfaction level</option>
                  <option value="1">Very Dissatisfied</option>
                  <option value="2">Dissatisfied</option>
                  <option value="3">Neutral</option>
                  <option value="4">Satisfied</option>
                  <option value="5">Very Satisfied</option>
                </select>
              </div>
              <div>
                <label htmlFor="financial_stress" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Financial Stress
                </label>
                <select
                  id="financial_stress"
                  name="financial_stress"
                  value={financialStress}
                  onChange={(e) => setFinancialStress(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select stress level</option>
                  <option value="1">Very Low</option>
                  <option value="2">Low</option>
                  <option value="3">Moderate</option>
                  <option value="4">High</option>
                  <option value="5">Very High</option>
                </select>
              </div>
            </div>
          </motion.div>

          {/* Lifestyle Information Section */}
          <motion.div variants={fadeIn} className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
            <div className="flex items-center gap-4 mb-6">
              <div className="p-3 bg-purple-100 dark:bg-purple-900 rounded-xl">
                <FiHeart className="w-6 h-6 text-purple-600 dark:text-purple-400" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Lifestyle Information</h2>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label htmlFor="sleep_duration" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Sleep Duration
                </label>
                <select
                  id="sleep_duration"
                  name="sleep_duration"
                  value={sleepDuration}
                  onChange={(e) => setSleepDuration(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select sleep duration</option>
                  <option value="<6 hours">Less than 6 hours</option>
                  <option value="6-7 hours">6-7 hours</option>
                  <option value="7-8 hours">7-8 hours</option>
                  <option value="8-9 hours">8-9 hours</option>
                  <option value=">9 hours">More than 9 hours</option>
                </select>
              </div>
              <div>
                <label htmlFor="dietary_habits" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Dietary Habits
                </label>
                <select
                  id="dietary_habits"
                  name="dietary_habits"
                  value={dietaryHabits}
                  onChange={(e) => setDietaryHabits(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select dietary habits</option>
                  <option value="Healthy">Healthy</option>
                  <option value="Moderate">Moderate</option>
                  <option value="Unhealthy">Unhealthy</option>
                </select>
              </div>
              <div>
                <label htmlFor="suicidal_thoughts" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Have you ever had suicidal thoughts?
                </label>
                <select
                  id="suicidal_thoughts"
                  name="suicidal_thoughts"
                  value={suicidalThoughts}
                  onChange={(e) => setSuicidalThoughts(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select an option</option>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
              <div>
                <label htmlFor="family_history" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Family History of Mental Illness
                </label>
                <select
                  id="family_history"
                  name="family_history"
                  value={familyHistory}
                  onChange={(e) => setFamilyHistory(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all duration-200"
                  required
                >
                  <option value="">Select an option</option>
                  <option value="Yes">Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>
          </motion.div>

          {/* Submit Button */}
          <motion.div variants={fadeIn} className="flex justify-center">
            <button
              type="submit"
              className="inline-flex items-center px-8 py-3 text-lg font-medium text-white bg-gradient-to-r from-indigo-600 to-blue-500 rounded-xl shadow-lg hover:from-indigo-700 hover:to-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transform transition-all duration-200 hover:scale-105 active:scale-95"
            >
              Analyze Risk Level
              <FiArrowRight className="ml-2 w-5 h-5" />
            </button>
          </motion.div>
        </motion.form>

        {/* Display Error Message */}
        {error && (
          <div className="mt-6 p-4 border border-red-300 bg-red-50 dark:border-red-700 dark:bg-red-900 rounded-md w-full text-center">
            <p className="text-sm font-medium text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {/* Prediction results */}
        {prediction && (
          <motion.div 
            id="prediction-box"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden"
          >
            {/* Header with gradient background */}
            <div className={`p-6 ${
              prediction.percentage >= 70 
                ? 'bg-gradient-to-r from-red-500 to-pink-500' 
                : prediction.percentage >= 40 
                  ? 'bg-gradient-to-r from-yellow-400 to-orange-500' 
                  : 'bg-gradient-to-r from-green-400 to-emerald-500'
            }`}>
              <h2 className="text-2xl font-bold text-white text-center">
                Risk Assessment Results
              </h2>
            </div>

            {/* Content */}
            <div className="p-8">
              {/* Risk Percentage */}
              <div className="text-center mb-8">
                <div className="inline-flex items-center justify-center w-32 h-32 rounded-full bg-gray-50 dark:bg-gray-700 mb-4">
                  <span className={`text-4xl font-bold ${
                    prediction.percentage >= 70 
                      ? 'text-red-500' 
                      : prediction.percentage >= 40 
                        ? 'text-yellow-500' 
                        : 'text-green-500'
                  }`}>
                    {prediction.percentage}%
                  </span>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">
                  {prediction.percentage >= 70 
                    ? 'High Risk Level' 
                    : prediction.percentage >= 40 
                      ? 'Moderate Risk Level' 
                      : 'Low Risk Level'}
                </h3>
              </div>

              {/* Progress Bar */}
              <div className="mb-8">
                <div className="h-3 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
                  <motion.div 
                    initial={{ width: 0 }}
                    animate={{ width: `${prediction.percentage}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                    className={`h-full rounded-full ${
                      prediction.percentage >= 70 
                        ? 'bg-gradient-to-r from-red-500 to-pink-500' 
                        : prediction.percentage >= 40 
                          ? 'bg-gradient-to-r from-yellow-400 to-orange-500' 
                          : 'bg-gradient-to-r from-green-400 to-emerald-500'
                    }`}
                  />
                </div>
              </div>

              {/* Message */}
              <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 mb-8">
                <p className="text-gray-700 dark:text-gray-300 text-center">
                  {prediction.message}
                </p>
              </div>

              {/* Disclaimer */}
              <div className="text-center mb-8">
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  This prediction is based on a statistical model and is not a substitute for professional medical advice. 
                  If you are concerned about your mental health, please consult a healthcare professional.
                </p>
              </div>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row justify-center gap-4">
                <Link 
                  href="/analysis" 
                  className="inline-flex items-center justify-center px-6 py-3 text-base font-medium text-white bg-gradient-to-r from-indigo-600 to-blue-500 rounded-xl shadow-lg hover:from-indigo-700 hover:to-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transform transition-all duration-200 hover:scale-105 active:scale-95"
                >
                  View Model Analysis
                  <FiArrowRight className="ml-2 w-5 h-5" />
                </Link>
              </div>
            </div>
          </motion.div>
        )}
      </main>
    </div>
  );
}
