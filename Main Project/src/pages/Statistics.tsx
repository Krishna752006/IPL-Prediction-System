import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, BarChart, Bar } from 'recharts';

const Statistics = () => {
  // Sample data - in a real application, this would come from your backend
  const winningTrends = [
    { year: '2020', CSK: 42.85, DC: 56.25, GT: null, KKR: 50, LSG: null, MI: 68.75, PBKS: 42.85, RR: 42.85, RCB: 46.66, SRH: 50 },
    { year: '2021', CSK: 68.75, DC: 62.5, GT: null, KKR: 52.94, LSG: null, MI: 50, PBKS: 42.85, RR: 35.71, RCB: 60, SRH: 21.42 },
    { year: '2022', CSK: 28.57, DC: 50, GT: 75, KKR: 42.85, LSG: 60, MI: 28.57, PBKS: 50, RR: 58.85, RCB: 56.25, SRH: 42.85 },
    { year: '2023', CSK: 62.5, DC: 35.71, GT: 64.7, KKR: 42.85, LSG: 53.33, MI: 56.25, PBKS: 42.85, RR: 50, RCB: 50, SRH: 28.67 },
    { year: '2024', CSK: 50, DC: 50, GT: 35.71, KKR: 68.75, LSG: 50, MI: 28.57, PBKS: 35.71, RR: 56.25, RCB: 53.33, SRH: 52.9 }
  ];
  
  const performanceStats = [
    { team: 'CSK', wins: 138, losses: 98, nrr: 0.169 },
    { team: 'DC', wins: 112, losses: 134, nrr: -0.089 },
    { team: 'GT', wins: 30, losses: 13, nrr: 0.396 },
    { team: 'KKR', wins: 130, losses: 117, nrr: 0.053 },
    { team: 'LSG', wins: 24, losses: 19, nrr: 0.116 },
    { team: 'MI', wins: 142, losses: 115, nrr: 0.105 },
    { team: 'PBKS', wins: 109, losses: 133, nrr: -0.099 },
    { team: 'RR', wins: 110, losses: 106, nrr: 0.018 },
    { team: 'RCB', wins: 121, losses: 128, nrr: -0.028 },
    { team: 'SRH', wins: 87, losses: 91, nrr: -0.023 },
  ];
  

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold text-gray-900">IPL Statistics Dashboard</h2>

      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-xl font-semibold mb-4">Team Performance Trends</h3>
        <div className="w-full overflow-x-auto">
          <LineChart width={800} height={400} data={winningTrends}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="year" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="CSK" stroke="#ffd700" /> {/* Chennai Super Kings */}
<Line type="monotone" dataKey="MI" stroke="#0066b2" /> {/* Mumbai Indians */}
<Line type="monotone" dataKey="RCB" stroke="#ff0000" /> {/* Royal Challengers Bangalore */}
<Line type="monotone" dataKey="KKR" stroke="#800080" /> {/* Kolkata Knight Riders */}
<Line type="monotone" dataKey="DC" stroke="#00a2ff" /> {/* Delhi Capitals */}
<Line type="monotone" dataKey="GT" stroke="#00ff7f" /> {/* Gujarat Titans */}
<Line type="monotone" dataKey="LSG" stroke="#ff69b4" /> {/* Lucknow Super Giants */}
<Line type="monotone" dataKey="PBKS" stroke="#a52a2a" /> {/* Punjab Kings */}
<Line type="monotone" dataKey="RR" stroke="#ff4500" /> {/* Rajasthan Royals */}
<Line type="monotone" dataKey="SRH" stroke="#ff8c00" /> {/* Sunrisers Hyderabad */}

          </LineChart>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-xl font-semibold mb-4">Overall Performance Statistics</h3>
        <div className="w-full overflow-x-auto">
          <BarChart width={800} height={400} data={performanceStats}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="team" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="wins" fill="#4CAF50" />
            <Bar dataKey="losses" fill="#f44336" />
          </BarChart>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-semibold mb-4">Key Insights</h3>
          <ul className="space-y-2">
            <li>• Mumbai Indians has the highest win percentage</li>
            <li>• CSK shows consistent performance across seasons</li>
            <li>• RCB shows improving trends in recent years</li>
            <li>• KKR maintains steady mid-table performance</li>
          </ul>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-semibold mb-4">Recent Trends</h3>
          <ul className="space-y-2">
            <li>• Increased importance of powerplay performance</li>
            <li>• Growing impact of death overs batting</li>
            <li>• Rise in successful chase completions</li>
            <li>• Higher average first innings scores</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default Statistics;