import { NextResponse } from 'next/server';

// This implements a simplified linear regression model to predict depression risk as a percentage
function predictDepressionRisk(data: any): { percentage: number, message: string } {
  // Base risk (intercept term in the regression)
  let riskPercentage = 30;
  
  // Sleep factor (lower sleep hours increase risk)
  const sleepHours = parseFloat(data.sleep);
  // Sleep has a negative coefficient (more sleep = lower risk)
  riskPercentage -= (sleepHours - 6) * 3; // For each hour above 6, reduce risk by 3%
  
  // Exercise factor (coefficients based on exercise frequency)
  if (data.exercise === 'never') riskPercentage += 15;
  else if (data.exercise === '1-2') riskPercentage += 5;
  else if (data.exercise === '3-4') riskPercentage -= 5;
  else if (data.exercise === '5+') riskPercentage -= 10;
  
  // Social interaction factor (coefficients based on social level)
  if (data.social === 'low') riskPercentage += 15;
  else if (data.social === 'medium') riskPercentage += 0;
  else if (data.social === 'high') riskPercentage -= 10;
  
  // Age factor
  const age = parseInt(data.age);
  // Simplified U-shaped relationship with age (higher risk for very young and older adults)
  if (age < 25) riskPercentage += 5;
  else if (age > 60) riskPercentage += 7;
  
  // Income factor
  if (data.income === '<30k') riskPercentage += 8;
  else if (data.income === '30k-60k') riskPercentage += 4;
  else if (data.income === '>150k') riskPercentage -= 3;
  
  // Ensure the percentage is within 0-100 range
  riskPercentage = Math.max(0, Math.min(100, riskPercentage));
  
  // Round to nearest integer
  riskPercentage = Math.round(riskPercentage);
  
  // Determine message based on risk percentage
  let message;
  if (riskPercentage >= 70) {
    message = "High Risk - You may be at increased risk for depression. Consider speaking with a healthcare professional.";
  } else if (riskPercentage >= 40) {
    message = "Moderate Risk - Some risk factors present. Regular self-care is recommended.";
  } else {
    message = "Low Risk - Few risk factors present. Continue maintaining healthy habits.";
  }
  
  return { percentage: riskPercentage, message };
}

export async function POST(request: Request) {
  try {
    // Parse the request body
    const data = await request.json();
    
    // Validate the input data (simplified validation)
    if (!data.age || !data.sleep || !data.exercise || !data.social) {
      return NextResponse.json(
        { error: 'Missing required fields' },
        { status: 400 }
      );
    }
    
    // Get prediction
    const prediction = predictDepressionRisk(data);
    
    // Return the prediction
    return NextResponse.json(prediction);
    
  } catch (error) {
    console.error('Error processing prediction:', error);
    return NextResponse.json(
      { error: 'Failed to process prediction' },
      { status: 500 }
    );
  }
} 