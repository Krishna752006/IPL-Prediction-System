import React, { useState } from 'react';
import { Users, Trophy, TrendingUp } from 'lucide-react';
import squadsData from '../data/ipl_2026_squads.json';
import winData from '../data/win_percentage_history.json';

const TeamAnalysis = () => {
  const [selectedTeamKey, setSelectedTeamKey] = useState('CSK');

  // Updated 2026 Meta: Sharp points, accurate key players based on new JSON roster
  const teamMeta = {
    'CSK': { 
      color: '#ffd700', 
      strengths: ['Solid Top Order', 'Elite Spin Attack', 'Strong Leadership'], 
      weaknesses: ['Pace Bowling Depth', 'Lower Middle-Order Experience'], 
      keyPlayers: ['Ruturaj Gaikwad', 'Sanju Samson', 'Noor Ahmad'] 
    },
    'MI': { 
      color: '#004ba0', 
      strengths: ['Core Indian Batters', 'Elite Death Bowling', 'Proven Leadership'], 
      weaknesses: ['Quality Spin Attack', 'Overseas Pacer Dependability'], 
      keyPlayers: ['Hardik Pandya', 'Jasprit Bumrah', 'Rohit Sharma', 'Surya Kumar Yadav'] 
    },
    'RCB': { 
      color: '#aa2020', 
      strengths: ['Star-Studded Top Order', 'Experienced Indian Pacers', 'Aggressive Batters'], 
      weaknesses: ['Spin Bowling Options', 'Death Bowling Consistency'], 
      keyPlayers: ['Virat Kohli', 'Phil Salt', 'Bhuvneshwar Kumar'] 
    },
    'DC': { 
      color: '#0047AB', 
      strengths: ['Explosive Openers', 'Elite Pace Attack', 'Top-Tier Spin Duo'], 
      weaknesses: ['Middle-Order Depth', 'Overseas Pacer Reliance'], 
      keyPlayers: ['Axar Patel', 'KL Rahul', 'Mitchell Starc', 'Kuldeep Yadav'] 
    },
    'KKR': { 
      color: '#3A225D', 
      strengths: ['World-Class All-Rounders', 'Deep Spin Attack', 'Dynamic Finishers'], 
      weaknesses: ['Pace Bowling Experience', 'Top-Order Stability'], 
      keyPlayers: ['Sunil Narine', 'Rinku Singh', 'Cameron Green', 'Ajinkya Rahane'] 
    },
    'PBKS': { 
      color: '#D71920', 
      strengths: ['Proven Match-Winners', 'Strong Spin Attack', 'Balanced All-Rounders'], 
      weaknesses: ['Lower-Order Finishing', 'Overseas Batting Depth'], 
      keyPlayers: ['Shreyas Iyer', 'Arshdeep Singh', 'Yuzvendra Chahal', 'Marcus Stoinis'] 
    },
    'RR': { 
      color: '#FF8C00', 
      strengths: ['Explosive Openers', 'Elite Pace Firepower', 'High-Quality Spinners'], 
      weaknesses: ['Backup Wicketkeepers', 'Lower-Order Power Hitting'], 
      keyPlayers: ['Yashasvi Jaiswal', 'Ravindra Jadeja', 'Jofra Archer', 'Ravi Bishnoi'] 
    },
    'SRH': { 
      color: '#F66000', 
      strengths: ['Devastating Openers', 'Fearless Middle Order', 'Captaincy & Strategy'], 
      weaknesses: ['Quality Indian Spinners', 'Heavy Overseas Reliance'], 
      keyPlayers: ['Pat Cummins', 'Travis Head', 'Heinrich Klaasen', 'Abhishek Sharma'] 
    },
    'LSG': { 
      color: '#F46100', 
      strengths: ['Aggressive Wicketkeepers', 'Express Pace Attack', 'Strong Middle Order'], 
      weaknesses: ['Quality Spin Depth', 'Opening Consistency'], 
      keyPlayers: ['Rishabh Pant', 'Nicholas Pooran', 'Mayank Yadav', 'Aiden Markram'] 
    },
    'GT': { 
      color: '#004ba0', 
      strengths: ['Formidable Openers', 'World-Class Spinners', 'Lethal Pace Attack'], 
      weaknesses: ['All-Rounder Depth', 'Middle-Order Stability'], 
      keyPlayers: ['Shubman Gill', 'Jos Buttler', 'Rashid Khan', 'Kagiso Rabada'] 
    }
  };

  // Pull dynamic data from JSON imports
  const currentSquad = squadsData.teams[selectedTeamKey as keyof typeof squadsData.teams];
  const currentWinStats = winData.find((team) => team.team.includes(selectedTeamKey));
  const currentMeta = teamMeta[selectedTeamKey as keyof typeof teamMeta];

  // Merge Playing XI and Bench, then strictly limit to a maximum of 25 players
  const playingXi = Object.values(currentSquad.playing_xi);
  const bench = currentSquad.bench || [];
  const allPlayers = [...playingXi, ...bench].slice(0, 25);
  
  const halfIndex = Math.ceil(allPlayers.length / 2);
  const firstHalf = allPlayers.slice(0, halfIndex);
  const secondHalf = allPlayers.slice(halfIndex);   

  return (
    <div className="space-y-8 animate-fadeIn">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between bg-white p-6 rounded-2xl shadow-sm border border-slate-100 gap-4">
        <div>
          <h2 className="text-2xl font-black text-slate-900 tracking-tight">Team Analysis</h2>
          <p className="text-sm text-slate-500">Deep dive into IPL 2026 squad dynamics</p>
        </div>
        <select
          value={selectedTeamKey}
          onChange={(e) => setSelectedTeamKey(e.target.value)}
          className="w-full sm:w-auto p-3 bg-slate-50 border border-slate-200 text-slate-800 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 font-semibold"
        >
          {winData.map((t: any) => {
            const acronym = t.team.match(/\(([^)]+)\)/)?.[1] || t.team;
            return (
              <option key={t.team} value={acronym}>
                {t.team}
              </option>
            );
          })}
        </select>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        <StatCard
          icon={<Trophy className="h-6 w-6" />}
          title="All-Time IPL Titles"
          value={currentWinStats?.titles || 0}
          color={currentMeta.color}
        />
        <StatCard
          icon={<TrendingUp className="h-6 w-6" />}
          title="Historical Win Rate"
          value={`${currentWinStats?.win_percentage || 0}%`}
          color={currentMeta.color}
        />
        <StatCard
          icon={<Users className="h-6 w-6" />}
          title="Current Squad Size"
          value={allPlayers.length} // Will cap at 25 based on the slice
          color={currentMeta.color}
        />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
          <h3 className="text-lg font-bold text-slate-900 mb-5">2026 Key Players</h3>
          <div className="space-y-4">
            {currentMeta.keyPlayers.map((player: string, index: number) => (
              <div key={index} className="flex items-center space-x-3 p-3 rounded-xl bg-slate-50 hover:bg-slate-100 transition-colors">
                <div className="w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: currentMeta.color }} />
                <span className="font-semibold text-slate-700">{player}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
          <h3 className="text-lg font-bold text-slate-900 mb-5">Tactical Analysis</h3>
          <div className="space-y-6">
            <div>
              <h4 className="text-xs font-black uppercase text-emerald-600 tracking-wider mb-3">Core Strengths</h4>
              <ul className="space-y-2">
                {currentMeta.strengths.map((strength: string, index: number) => (
                  <li key={index} className="flex items-center text-sm font-medium text-slate-700">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 mr-2.5" />
                    {strength}
                  </li>
                ))}
              </ul>
            </div>
            <div className="border-t border-slate-100 pt-5">
              <h4 className="text-xs font-black uppercase text-rose-600 tracking-wider mb-3">Vulnerabilities</h4>
              <ul className="space-y-2">
                {currentMeta.weaknesses.map((weakness: string, index: number) => (
                  <li key={index} className="flex items-center text-sm font-medium text-slate-700">
                    <span className="w-1.5 h-1.5 rounded-full bg-rose-500 mr-2.5" />
                    {weakness}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-bold text-slate-900">Official Squad List (2026)</h3>
          <span className="text-xs font-bold bg-blue-50 text-blue-600 px-3 py-1 rounded-lg border border-blue-100">
            {allPlayers.length}/25 Players
          </span>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
          <ul className="space-y-2">
            {firstHalf.map((player: any, index: number) => (
              <li key={index} className="flex items-center text-sm font-medium text-slate-700 p-2 rounded-lg hover:bg-slate-50">
                <span className="w-6 text-slate-400 text-xs">{index + 1}.</span>
                {player.name} 
                {player.overseas && <span className="ml-2 text-[10px] bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded font-bold tracking-wider">OS</span>}
              </li>
            ))}
          </ul>
          <ul className="space-y-2">
            {secondHalf.map((player: any, index: number) => (
              <li key={index} className="flex items-center text-sm font-medium text-slate-700 p-2 rounded-lg hover:bg-slate-50">
                <span className="w-6 text-slate-400 text-xs">{halfIndex + index + 1}.</span>
                {player.name} 
                {player.overseas && <span className="ml-2 text-[10px] bg-slate-200 text-slate-600 px-1.5 py-0.5 rounded font-bold tracking-wider">OS</span>}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ icon, title, value, color }: any) => {
  return (
    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100 hover:shadow-md transition-shadow relative overflow-hidden group">
      <div className="absolute -right-6 -top-6 opacity-10 group-hover:scale-110 transition-transform duration-300" style={{ color }}>
        {React.cloneElement(icon, { className: 'w-32 h-32' })}
      </div>
      <div className="relative z-10 flex flex-col h-full justify-between">
        <div className="flex items-center gap-3 mb-4">
          <div className="p-3 rounded-xl bg-slate-50" style={{ color }}>{icon}</div>
          <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider">{title}</h3>
        </div>
        <span className="text-4xl font-black" style={{ color }}>{value}</span>
      </div>
    </div>
  );
};

export default TeamAnalysis;