import React, { useState, useEffect } from 'react';
import { History, Loader2, AlertCircle } from 'lucide-react';
import { useAuthStore } from '../store/authStore';

interface PredictionRecord {
  id?: string;
  team_a: string;
  team_b: string;
  venue: string;
  winner: string;
  result: string;
  predicted_at: string;
}

const PredictionHistory: React.FC = () => {
  // Adjust 'token' if your store uses a different naming convention
  const { user, token } = useAuthStore(); 
  const [history, setHistory] = useState<PredictionRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        if (user?.isGuest) {
          // Fetch guest data from local storage
          const existing = JSON.parse(localStorage.getItem('predictionHistory') || '[]');
          setHistory(existing);
        } else {
          // Fetch authenticated user data from backend
          const response = await fetch('http://localhost:8000/prediction-history', {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token}` 
            }
          });

          if (!response.ok) {
            throw new Error('Failed to fetch prediction history');
          }
          
          const data = await response.json();
          setHistory(data);
        }
      } catch (err) {
        setError('Could not load prediction history. Please try again later.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [user, token]);

  // Helper to format the ISO date string
  const formatDate = (dateString: string) => {
    const options: Intl.DateTimeFormatOptions = { 
      year: 'numeric', month: 'short', day: 'numeric', 
      hour: '2-digit', minute: '2-digit' 
    };
    return new Date(dateString).toLocaleDateString(undefined, options);
  };

  return (
    <div className="max-w-6xl mx-auto py-8 px-4">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-center gap-3 mb-8">
          <History className="h-10 w-10 text-indigo-600" />
          <h2 className="text-3xl font-bold text-gray-800">Your Prediction History</h2>
        </div>

        {loading ? (
          <div className="flex justify-center items-center py-12">
            <Loader2 className="animate-spin h-8 w-8 text-indigo-600" />
            <span className="ml-3 text-lg text-gray-600">Loading history...</span>
          </div>
        ) : error ? (
          <div className="flex items-center gap-2 text-red-600 bg-red-50 p-4 rounded-md">
            <AlertCircle className="h-5 w-5" />
            <p>{error}</p>
          </div>
        ) : history.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <p className="text-lg">No predictions found.</p>
            <p className="mt-2">Head over to the Predictions page to simulate your first match!</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full border-collapse bg-white text-left text-sm text-gray-500">
              <thead className="bg-gray-100">
                <tr>
                  <th className="px-6 py-4 font-medium text-gray-900">Date</th>
                  <th className="px-6 py-4 font-medium text-gray-900">Matchup</th>
                  <th className="px-6 py-4 font-medium text-gray-900">Venue</th>
                  <th className="px-6 py-4 font-medium text-gray-900">Predicted Winner</th>
                  <th className="px-6 py-4 font-medium text-gray-900">Result Details</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 border-t border-gray-200">
                {history.map((record, index) => (
                  <tr key={record.id || index} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap">
                      {formatDate(record.predicted_at)}
                    </td>
                    <td className="px-6 py-4 font-medium text-gray-900">
                      {record.team_a} <span className="text-gray-400 mx-1">vs</span> {record.team_b}
                    </td>
                    <td className="px-6 py-4">
                      {record.venue || 'Random Venue'}
                    </td>
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center gap-1 rounded-full bg-green-50 px-2 py-1 text-xs font-semibold text-green-600">
                        {record.winner}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-600">
                      {record.result}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionHistory;