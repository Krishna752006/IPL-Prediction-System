const BASE_URL = 'http://localhost:5000/api';

interface PredictionResult {
  teamA: {
    score: number;
    wickets: number;
    run_rate: number;
  };
  teamB: {
    score: number;
    wickets: number;
    run_rate: number;
    match_result: string;
  };
}

const PredictionModel = {
  predict: async (teamA: number, teamB: number, venue: number): Promise<PredictionResult | null> => {
    // Check if teamA and teamB are the same
    if (teamA === teamB) {
      console.error('Error: Please provide two different teams as input.');
      return null;
    }

    try {
      const response = await fetch(`${BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ teamA, teamB, venue }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction');
      }

      return (await response.json()) as PredictionResult;
    } catch (error) {
      console.error('Prediction Error:', error);
      return null;
    }
  },
};

export default PredictionModel;
