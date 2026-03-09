import React, { useState, useEffect } from 'react';
import { Brain } from 'lucide-react';
import PredictionModel from '../utils/PredictionModel';

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

const Predictions: React.FC = () => {
  const [team1, setTeam1] = useState<number | null>(null);
  const [team2, setTeam2] = useState<number | null>(null);
  const [venue, setVenue] = useState<number | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPredictDisabled, setIsPredictDisabled] = useState<boolean>(false);
  const [countdown, setCountdown] = useState<number>(0);

  const teams = [
    { id: 1, name: 'Chennai Super Kings' },
    { id: 2, name: 'Delhi Capitals' },
    { id: 5, name: 'Kolkata Knight Riders' },
    { id: 7, name: 'Mumbai Indians' },
    { id: 9, name: 'Punjab Kings' },
    { id: 10, name: 'Rajasthan Royals' },
    { id: 12, name: 'Royal Challengers Bangalore' },
    { id: 13, name: 'Sunrisers Hyderabad' },
    { id: 3, name: 'Gujarat Titans' },
    { id: 6, name: 'Lucknow Super Giants' }
  ];

  const venues = [
    { id: 1, name: 'Arun Jaitley Stadium' },
    { id: 3, name: 'Barsapara Cricket Stadium' },
    { id: 7, name: 'Dr DY Patil Sports Academy' },
    { id: 10, name: 'Eden Gardens' },
    { id: 11, name: 'Ekana Cricket Stadium' },
    { id: 12, name: 'Feroz Shah Kotla' },
    { id: 18, name: 'M Chinnaswamy Stadium' },
    { id: 19, name: 'MA Chidambaram Stadium' },
    { id: 20, name: 'Maharaja Yadavindra Singh International Cricket Stadium' },
    { id: 22, name: 'Narendra Modi Stadium' },
    { id: 27, name: 'Punjab Cricket Association IS Bindra Stadium' },
    { id: 28, name: 'Punjab Cricket Association Stadium' },
    { id: 29, name: 'Rajiv Gandhi International Stadium' },
    { id: 30, name: 'Sardar Patel Stadium' },
    { id: 31, name: 'Saurashtra Cricket Association Stadium' },
    { id: 32, name: 'Sawai Mansingh Stadium' },
    { id: 35, name: 'Sheikh Zayed Stadium' },
    { id: 37, name: 'Subrata Roy Sahara Stadium' },
    { id: 39, name: 'Vidarbha Cricket Association Stadium' },
    { id: 40, name: 'Wankhede Stadium' }
  ];

  const handlePredict = async () => {
    if (team1 === team2) {
      setErrorMessage('Please give 2 different teams as input.');
      return;
    }
    setErrorMessage(null);

    if (team1 !== null && team2 !== null && venue !== null) {
      const result = await PredictionModel.predict(team1, team2, venue);
      setPrediction(result);
      setIsPredictDisabled(true);
      setCountdown(100);
    }
  };

  useEffect(() => {
    if (countdown > 0) {
      const timerId = setInterval(() => setCountdown((prev) => prev - 1), 1000);
      return () => clearInterval(timerId);
    } else {
      setIsPredictDisabled(false);
    }
  }, [countdown]);

  const getTeamName = (teamId: number | null) => {
    return teams.find((team) => team.id === teamId)?.name || 'Unknown Team';
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center mb-6">
          <Brain className="h-12 w-12 text-indigo-600" />
        </div>

        <h2 className="text-2xl font-bold text-center mb-6">Match Prediction</h2>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Team 1</label>
            <select
              value={team1 || ''}
              onChange={(e) => setTeam1(Number(e.target.value))}
              className="w-full p-2 border rounded-md"
            >
              <option value="">Select Team 1</option>
              {teams.map((team) => (
                <option key={team.id} value={team.id}>{team.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Team 2</label>
            <select
              value={team2 || ''}
              onChange={(e) => setTeam2(Number(e.target.value))}
              className="w-full p-2 border rounded-md"
            >
              <option value="">Select Team 2</option>
              {teams.map((team) => (
                <option key={team.id} value={team.id}>{team.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Venue</label>
            <select
              value={venue || ''}
              onChange={(e) => setVenue(Number(e.target.value))}
              className="w-full p-2 border rounded-md"
            >
              <option value="">Select Venue</option>
              {venues.map((v) => (
                <option key={v.id} value={v.id}>{v.name}</option>
              ))}
            </select>
          </div>

          <button
            onClick={handlePredict}
            disabled={isPredictDisabled || team1 === null || team2 === null || venue === null}
            className="w-full bg-indigo-600 text-white py-2 px-4 rounded-md hover:bg-indigo-700 disabled:bg-gray-400"
          >
            {isPredictDisabled ? `Wait ${countdown}s` : 'Predict Winner'}
          </button>
        </div>

        {errorMessage && (
          <div className="mt-4 text-red-500 font-medium">{errorMessage}</div>
        )}

        {prediction && (
          <div className="mt-6 p-4 bg-gray-50 rounded-md">
            <h3 className="text-lg font-semibold mb-2">Prediction Result:</h3>
            <p><strong>{getTeamName(team1)}:</strong> {prediction.teamA.score} runs, {prediction.teamA.wickets} wickets, Run Rate: {prediction.teamA.run_rate}</p>
            <p><strong>{getTeamName(team2)}:</strong> {prediction.teamB.score} runs, {prediction.teamB.wickets} wickets, Run Rate: {prediction.teamB.run_rate}</p>
            <p><strong>Match Result:</strong> {prediction.teamB.match_result}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Predictions;
