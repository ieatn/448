import { NextResponse } from 'next/server';

// This is a simplified model that returns a prediction based on the inputs
// In a real application, this would use an actual trained model
function predictDepressionRisk(data: any): string {
  // Calculate a simple risk score based on the input factors
  let riskScore = 0;
  
  // Sleep factor (lower sleep hours increase risk)
  const sleepHours = parseFloat(data.sleep);
  if (sleepHours < 6) riskScore += 3;
  else if (sleepHours < 7) riskScore += 2;
  else if (sleepHours < 8) riskScore += 1;
  
  // Exercise factor
  if (data.exercise === 'never') riskScore += 3;
  else if (data.exercise === '1-2') riskScore += 2;
  else if (data.exercise === '3-4') riskScore += 1;
  
  // Social interaction factor
  if (data.social === 'low') riskScore += 3;
  else if (data.social === 'medium') riskScore += 1;
  
  // Age factor (simplified)
  const age = parseInt(data.age);
  if (age > 60) riskScore += 1;
  else if (age < 25) riskScore += 1;
  
  // Income factor (simplified)
  if (data.income === '<30k') riskScore += 1;
  
  // Determine risk category
  if (riskScore >= 7) {
    return "High Risk - You may be at increased risk for depression. Consider speaking with a healthcare professional.";
  } else if (riskScore >= 4) {
    return "Moderate Risk - Some risk factors present. Regular self-care is recommended.";
  } else {
    return "Low Risk - Few risk factors present. Continue maintaining healthy habits.";
  }
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
    return NextResponse.json({ prediction });
    
  } catch (error) {
    console.error('Error processing prediction:', error);
    return NextResponse.json(
      { error: 'Failed to process prediction' },
      { status: 500 }
    );
  }
} 