'use client';

import { useState } from 'react';
import Link from 'next/link';

interface PredictionResult {
  percentage: number;
  message: string;
}

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

    } catch (err) {
      console.error('Failed to fetch prediction:', err);
      setError(err instanceof Error ? err.message : 'Failed to get prediction. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-8 font-[family-name:var(--font-geist-sans)] dark:bg-gray-900 dark:text-white">
      <main className="flex flex-col gap-8 items-center w-full max-w-6xl bg-white dark:bg-gray-800 p-6 rounded-lg shadow-md">
        <h1 className="text-2xl font-semibold text-center">Student Depression Risk Assessment</h1>
        
        {/* <div className="w-full text-right flex gap-2 justify-end">
          <button
            onClick={testServer}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-green-600 hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500"
          >
            Test Server Connection
          </button>
          <Link 
            href="/analysis" 
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            View Model Analysis
          </Link>
        </div> */}

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
              <option value="Male">Male</option>
              <option value="Female">Female</option>
              <option value="Other">Other</option>
            </select>
          </div>

          {/* Profession Input */}
          <div>
            <label htmlFor="profession" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Profession</label>
            <input
              type="text"
              id="profession"
              value={profession}
              onChange={(e) => setProfession(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Academic Pressure */}
          <div>
            <label htmlFor="academicPressure" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Academic Pressure (1-5)</label>
            <input
              type="number"
              id="academicPressure"
              min="1"
              max="5"
              value={academicPressure}
              onChange={(e) => setAcademicPressure(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Work Pressure */}
          <div>
            <label htmlFor="workPressure" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Work Pressure (1-5)</label>
            <input
              type="number"
              id="workPressure"
              min="1"
              max="5"
              value={workPressure}
              onChange={(e) => setWorkPressure(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* CGPA */}
          <div>
            <label htmlFor="cgpa" className="block text-sm font-medium text-gray-700 dark:text-gray-300">CGPA (0-10)</label>
            <input
              type="number"
              id="cgpa"
              min="0"
              max="10"
              step="0.1"
              value={cgpa}
              onChange={(e) => setCgpa(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Study Satisfaction */}
          <div>
            <label htmlFor="studySatisfaction" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Study Satisfaction (1-5)</label>
            <input
              type="number"
              id="studySatisfaction"
              min="1"
              max="5"
              value={studySatisfaction}
              onChange={(e) => setStudySatisfaction(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Job Satisfaction */}
          <div>
            <label htmlFor="jobSatisfaction" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Job Satisfaction (1-5)</label>
            <input
              type="number"
              id="jobSatisfaction"
              min="1"
              max="5"
              value={jobSatisfaction}
              onChange={(e) => setJobSatisfaction(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Sleep Duration */}
          <div>
            <label htmlFor="sleepDuration" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Sleep Duration</label>
            <select
              id="sleepDuration"
              value={sleepDuration}
              onChange={(e) => setSleepDuration(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Sleep Duration</option>
              <option value="<6 hours">Less than 6 hours</option>
              <option value="6-7 hours">6-7 hours</option>
              <option value="7-8 hours">7-8 hours</option>
              <option value="8-9 hours">8-9 hours</option>
              <option value=">9 hours">More than 9 hours</option>
            </select>
          </div>

          {/* Dietary Habits */}
          <div>
            <label htmlFor="dietaryHabits" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Dietary Habits</label>
            <select
              id="dietaryHabits"
              value={dietaryHabits}
              onChange={(e) => setDietaryHabits(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Dietary Habits</option>
              <option value="Healthy">Healthy</option>
              <option value="Moderate">Moderate</option>
              <option value="Unhealthy">Unhealthy</option>
            </select>
          </div>

          {/* Degree */}
          <div>
            <label htmlFor="degree" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Degree</label>
            <select
              id="degree"
              value={degree}
              onChange={(e) => setDegree(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Degree</option>
              <option value="BSc">BSc</option>
              <option value="MSc">MSc</option>
              <option value="PhD">PhD</option>
              <option value="Other">Other</option>
            </select>
          </div>

          {/* Suicidal Thoughts */}
          <div>
            <label htmlFor="suicidalThoughts" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Have you ever had suicidal thoughts?</label>
            <select
              id="suicidalThoughts"
              value={suicidalThoughts}
              onChange={(e) => setSuicidalThoughts(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Answer</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>

          {/* Work/Study Hours */}
          <div>
            <label htmlFor="workStudyHours" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Work/Study Hours per Day</label>
            <input
              type="number"
              id="workStudyHours"
              min="0"
              max="24"
              value={workStudyHours}
              onChange={(e) => setWorkStudyHours(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Financial Stress */}
          <div>
            <label htmlFor="financialStress" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Financial Stress (1-5)</label>
            <input
              type="number"
              id="financialStress"
              min="1"
              max="5"
              value={financialStress}
              onChange={(e) => setFinancialStress(e.target.value)}
              required
              className="mt-1 block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            />
          </div>

          {/* Family History */}
          <div>
            <label htmlFor="familyHistory" className="block text-sm font-medium text-gray-700 dark:text-gray-300">Family History of Mental Illness</label>
            <select
              id="familyHistory"
              value={familyHistory}
              onChange={(e) => setFamilyHistory(e.target.value)}
              required
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 dark:border-gray-600 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              <option value="" disabled>Select Answer</option>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>

          <div className="flex gap-2">
            <button
              type="button"
              onClick={fillSampleData}
              className="w-full inline-flex justify-center py-2 px-4 border border-indigo-600 shadow-sm text-sm font-medium rounded-md text-indigo-600 bg-white hover:bg-indigo-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Fill with Low Risk Sample
            </button>

            <button
              type="button"
              onClick={fillDepressionSampleData}
              className="w-full inline-flex justify-center py-2 px-4 border border-red-600 shadow-sm text-sm font-medium rounded-md text-red-600 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
            >
              Fill with High Risk Sample
            </button>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="mt-4 w-full inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? 'Getting Prediction...' : 'Get Prediction'}
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
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">Depression Risk Prediction:</h2>
            <div className="my-4">
              <div className="w-full bg-gray-200 dark:bg-gray-600 rounded-full h-4">
                <div 
                  className={`h-4 rounded-full ${
                    prediction.percentage >= 70 
                      ? 'bg-red-600' 
                      : prediction.percentage >= 40 
                        ? 'bg-yellow-500' 
                        : 'bg-green-500'
                  }`} 
                  style={{ width: `${prediction.percentage}%` }}
                ></div>
              </div>
              <p className="mt-2 font-bold text-lg">{prediction.percentage}% Risk</p>
            </div>
            <p className="mt-2 text-gray-700 dark:text-gray-300">{prediction.message}</p>
            <p className="mt-4 text-xs text-gray-500 dark:text-gray-400">Disclaimer: This prediction is based on a statistical model and is not a substitute for professional medical advice. If you are concerned about your mental health, please consult a healthcare professional.</p>
            
            <div className="mt-6">
              <Link 
                href="/analysis" 
                className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
              >
                View Model Analysis
              </Link>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
