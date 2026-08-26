import React from 'react';
import { RefreshCw, Trophy, Sparkles } from 'lucide-react';
import scheduleData from '../data/schedule.json';
import { usePredictionStore } from '../store/predictionStore';

const Schedule = () => {
  const { predictions, isPredicting, predictAllMatches, clearPredictions } = usePredictionStore();
  const matches = Object.entries(scheduleData);
  const totalPredicted = Object.keys(predictions).length;

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header & Controls */}
      <div className="bg-white rounded-xl shadow-md p-6 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">IPL 2026 Schedule</h1>
          <p className="text-sm text-gray-500 mt-1">
            {totalPredicted > 0 
              ? `Predicted ${totalPredicted} league matches.` 
              : 'Click "Predict All" to generate AI match predictions & standings.'}
          </p>
        </div>

        <div className="flex items-center space-x-3">
          {totalPredicted > 0 && (
            <button
              onClick={clearPredictions}
              className="flex items-center space-x-1 px-4 py-2 border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
            >
              <RefreshCw className="h-4 w-4" />
              <span>Reset</span>
            </button>
          )}

          <button
            onClick={predictAllMatches}
            disabled={isPredicting}
            className="flex items-center space-x-2 px-5 py-2.5 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:opacity-50 shadow-md transition-all"
          >
            {isPredicting ? (
              <>
                <RefreshCw className="h-5 w-5 animate-spin" />
                <span>Predicting...</span>
              </>
            ) : (
              <>
                <Sparkles className="h-5 w-5 text-yellow-300" />
                <span>{totalPredicted > 0 ? 'Re-predict All' : 'Predict All Matches'}</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Schedule & Prediction Results Table */}
      <div className="bg-white shadow rounded-xl overflow-hidden border border-gray-100">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Match
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Fixture
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Venue
              </th>
              <th className="px-6 py-3 text-center text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Predicted Winner
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {matches.map(([matchNo, details]) => {
              const prediction = predictions[matchNo];
              const matchTitle = details[0];
              const venue = details[1];

              return (
                <tr key={matchNo} className="hover:bg-indigo-50/30 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 font-medium">
                    {isNaN(Number(matchNo)) ? matchNo : `Match ${matchNo}`}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-semibold text-gray-900">
                    {matchTitle || 'Playoff Fixture (TBD)'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    {venue}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm">
                    {prediction ? (
                      <span className="inline-flex items-center space-x-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
                        <Trophy className="h-3.5 w-3.5 text-emerald-600" />
                        <span>{prediction.winner}</span>
                        <span className="text-emerald-600 opacity-80">({prediction.confidence}%)</span>
                      </span>
                    ) : (
                      <span className="text-gray-400 text-xs italic">Not predicted</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default Schedule;