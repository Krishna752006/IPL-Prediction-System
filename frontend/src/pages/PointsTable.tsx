import React from 'react';
import { usePredictionStore } from '../store/predictionStore';
import { Trophy, Award } from 'lucide-react';
import { Link } from 'react-router-dom';

const PointsTable = () => {
  const { getPointsTable, predictions } = usePredictionStore();
  const standings = getPointsTable();
  const hasPredictions = Object.keys(predictions).length > 0;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white p-6 rounded-xl shadow-md">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">IPL 2026 Standings</h1>
          <p className="text-sm text-gray-500 mt-1">
            {hasPredictions
              ? 'Standings automatically calculated from predicted schedule outcomes.'
              : 'No predictions generated yet. Run "Predict All" on the Schedule page!'}
          </p>
        </div>
        {!hasPredictions && (
          <Link
            to="/schedule"
            className="px-4 py-2 bg-indigo-600 text-white text-sm font-medium rounded-lg hover:bg-indigo-700 transition-colors shadow"
          >
            Go to Schedule
          </Link>
        )}
      </div>

      <div className="bg-white shadow rounded-xl overflow-hidden border border-gray-100">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Pos
              </th>
              <th className="px-6 py-3 text-left text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Team
              </th>
              <th className="px-6 py-3 text-center text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Played (P)
              </th>
              <th className="px-6 py-3 text-center text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Won (W)
              </th>
              <th className="px-6 py-3 text-center text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Lost (L)
              </th>
              <th className="px-6 py-3 text-center text-xs font-semibold text-gray-500 uppercase tracking-wider">
                Points (PTS)
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {standings.map((team, index) => {
              const isPlayoffSpot = index < 4;

              return (
                <tr
                  key={team.team}
                  className={`hover:bg-gray-50 transition-colors ${
                    isPlayoffSpot ? 'bg-indigo-50/20' : ''
                  }`}
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-700">
                    <span className="flex items-center space-x-2">
                      <span>{index + 1}</span>
                      {index === 0 && <Trophy className="h-4 w-4 text-yellow-500" />}
                      {isPlayoffSpot && index > 0 && (
                        <Award className="h-4 w-4 text-indigo-400" />
                      )}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">
                    {team.team}
                    {isPlayoffSpot && (
                      <span className="ml-2 text-[10px] px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded-full font-medium">
                        Playoffs
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm text-gray-600">
                    {team.played}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm font-semibold text-emerald-600">
                    {team.won}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm font-semibold text-rose-500">
                    {team.lost}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-center text-sm font-extrabold text-indigo-600 bg-indigo-50/50">
                    {team.points}
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

export default PointsTable;