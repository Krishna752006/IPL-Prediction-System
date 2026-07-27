import React, { useState } from 'react';
import { 
  Trophy, 
  Swords, 
  TrendingUp, 
  Target, 
  Award, 
  ShieldAlert, 
  Flame, 
  BarChart2
} from 'lucide-react';

// Import datasets
import highestLowest from '../data/highest,lowest.json';
import lowestDefended from '../data/lowest_score_defended_all_time.json';
import highestRunScorer from '../data/highest_run_scorer.json';
import eachTeamHighestLowest from '../data/each_team_highest_and_lowest.json';
import headToHead from '../data/head_to_head.json';
import highestRunsPlayerVsTeam from '../data/highest_runs_by_player_of_each_team_against_each_team.json';
import winPercentageHistory from '../data/win_percentage_history.json';

// Define explicit types to fix TS7053 implicit 'any' indexing errors
type TeamScoreData = Record<string, Record<string, { highest: string; lowest: string }>>;
type PlayerScoreData = Record<string, Record<string, { player: string; score: string; year: string }>>;

const Statistics: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'headToHead' | 'overview' | 'franchises' | 'records'>('overview');
  
  // State for Head-to-Head matchup tool
  const [team1, setTeam1] = useState<string>('CSK');
  const [team2, setTeam2] = useState<string>('MI');

  // Teams list for dropdown selectors
  const teamList = ['CSK', 'MI', 'RCB', 'KKR', 'DC', 'RR', 'PBKS', 'SRH', 'GT', 'LSG'];

  // Helper: Find Head-to-Head data
  const h2hRecord = headToHead.find(
    (item: any) => 
      (item.team1 === team1 && item.team2 === team2) || 
      (item.team1 === team2 && item.team2 === team1)
  );

  // Safely cast JSON objects to generic Records to allow dynamic template literal indexing
  const allTeamScores: TeamScoreData = eachTeamHighestLowest.highest_and_lowest_scores_against_each_current_team_till_2025;
  const allPlayerScores: PlayerScoreData = highestRunsPlayerVsTeam.highest_individual_score_by_player_for_each_team_against_each_opponent;

  // Helper: Find Matchup Scores
  const t1VsT2Scores = allTeamScores[team1]?.[`vs_${team2}`];
  const t2VsT1Scores = allTeamScores[team2]?.[`vs_${team1}`];

  // Helper: Find Top Individual Batting Performers for Matchup
  const t1PlayerVsT2 = allPlayerScores[team1]?.[`vs_${team2}`];
  const t2PlayerVsT1 = allPlayerScores[team2]?.[`vs_${team1}`];

  // All-time highest score calculation
  const highestMatch = [...highestLowest].sort((a: any, b: any) => {
    return parseInt(b.highest.score.split('/')[0]) - parseInt(a.highest.score.split('/')[0]);
  })[0];

  // All-time lowest score calculation
  const lowestMatch = [...highestLowest].sort((a: any, b: any) => {
    return parseInt(a.lowest.score.split('/')[0]) - parseInt(b.lowest.score.split('/')[0]);
  })[0];

  const lowestDefendedMatch = lowestDefended.lowest_scores_defended_in_ipl_top_10[0];

  const topRunScorer = Object.entries(
    highestRunScorer.highest_run_scorer_for_each_team
  ).sort((a: any, b: any) => Number(b[1].runs) - Number(a[1].runs))[0];

  return (
    <div className="max-w-7xl mx-auto space-y-8 px-2 sm:px-4">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-indigo-950 to-blue-900 rounded-3xl p-8 text-white shadow-2xl relative overflow-hidden">
        <div className="absolute right-0 top-0 translate-x-12 -translate-y-12 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-6">
          <div>
            <div className="flex items-center gap-2 text-blue-400 font-semibold mb-2 text-sm uppercase tracking-wider">
              <BarChart2 className="w-4 h-4" /> Comprehensive Data Intelligence
            </div>
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight">IPL Historical Analytics</h1>
            <p className="text-slate-300 mt-2 text-sm sm:text-base max-w-xl">
              In-depth franchise head-to-heads, individual records, match extremes, and all-time tournament statistics.
            </p>
          </div>
          
          <div className="flex items-center bg-slate-800/80 backdrop-blur-md px-4 py-2 rounded-2xl border border-slate-700/60 shadow-inner">
            <Trophy className="w-5 h-5 text-yellow-400 mr-2" />
            <span className="text-xs sm:text-sm font-medium text-slate-200">IPL 2026 Ready</span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex flex-wrap gap-2 mt-8 pt-6 border-t border-slate-800">
          {[
            { id: 'overview', label: 'Overview Highlights', icon: Flame },
            { id: 'headToHead', label: 'Rivalry & H2H Matrix', icon: Swords },
            { id: 'franchises', label: 'Franchise History', icon: Trophy },
            { id: 'records', label: 'All-Time Records', icon: Target },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-medium text-sm transition-all duration-200 ${
                  isActive
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/30'
                    : 'bg-slate-800/60 hover:bg-slate-800 text-slate-300 hover:text-white'
                }`}
              >
                <Icon className="w-4 h-4" />
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* ================= TABS CONTENT ================= */}

      {/* TAB 1: OVERVIEW HIGHLIGHTS */}
      {activeTab === 'overview' && (
        <div className="space-y-8 animate-fadeIn">
          {/* Key Metric Hero Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Card 1 */}
            <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100 hover:shadow-xl transition-shadow relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-4 text-emerald-500 opacity-20 group-hover:opacity-100 transition-opacity">
                <TrendingUp className="w-10 h-10" />
              </div>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Highest Team Score</span>
              <h3 className="text-3xl font-black text-emerald-600 mt-2">{highestMatch.highest.score}</h3>
              <p className="font-bold text-slate-800 mt-2">{highestMatch.team}</p>
              <p className="text-xs text-slate-500">vs {highestMatch.highest.opposition}</p>
            </div>

            {/* Card 2 */}
            <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100 hover:shadow-xl transition-shadow relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-4 text-rose-500 opacity-20 group-hover:opacity-100 transition-opacity">
                <ShieldAlert className="w-10 h-10" />
              </div>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Lowest Team Score</span>
              <h3 className="text-3xl font-black text-rose-600 mt-2">{lowestMatch.lowest.score}</h3>
              <p className="font-bold text-slate-800 mt-2">{lowestMatch.team}</p>
              <p className="text-xs text-slate-500">vs {lowestMatch.lowest.opposition}</p>
            </div>

            {/* Card 3 */}
            <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100 hover:shadow-xl transition-shadow relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-4 text-indigo-500 opacity-20 group-hover:opacity-100 transition-opacity">
                <Award className="w-10 h-10" />
              </div>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Lowest Defended</span>
              <h3 className="text-3xl font-black text-indigo-600 mt-2">{lowestDefendedMatch.score}</h3>
              <p className="font-bold text-slate-800 mt-2">{lowestDefendedMatch.team}</p>
              <p className="text-xs text-slate-500">vs {lowestDefendedMatch.opponent} ({lowestDefendedMatch.year})</p>
            </div>

            {/* Card 4 */}
            <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100 hover:shadow-xl transition-shadow relative overflow-hidden group">
              <div className="absolute top-0 right-0 p-4 text-amber-500 opacity-20 group-hover:opacity-100 transition-opacity">
                <Trophy className="w-10 h-10" />
              </div>
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">Top Run Scorer</span>
              <h3 className="text-3xl font-black text-amber-500 mt-2">{topRunScorer[1].runs} <span className="text-xs font-normal text-slate-400">runs</span></h3>
              <p className="font-bold text-slate-800 mt-2">{topRunScorer[1].player}</p>
              <p className="text-xs text-slate-500">{topRunScorer[0]} • {topRunScorer[1].matches} matches</p>
            </div>
          </div>

          {/* Featured Highlights Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* All Time Lowest Scores Defended */}
            <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 bg-indigo-50 text-indigo-600 rounded-xl">
                    <ShieldAlert className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-slate-900">Lowest Scores Defended</h3>
                    <p className="text-xs text-slate-500">Top 6 defend records in IPL history</p>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                {lowestDefended.lowest_scores_defended_in_ipl_top_10.map((item: any, idx: number) => (
                  <div key={idx} className="flex items-center justify-between p-3.5 rounded-xl bg-slate-50 hover:bg-indigo-50/50 transition-colors border border-slate-100">
                    <div className="flex items-center gap-3">
                      <span className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-xs ${
                        idx === 0 ? 'bg-amber-400 text-slate-900' : 'bg-slate-200 text-slate-700'
                      }`}>
                        #{item.rank}
                      </span>
                      <div>
                        <p className="font-semibold text-slate-800 text-sm">{item.team} vs {item.opponent}</p>
                        <p className="text-xs text-slate-500">{item.result}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className="font-extrabold text-indigo-600">{item.score}</span>
                      <p className="text-[10px] text-slate-400">{item.year}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Franchise Leaders Overview */}
            <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 bg-amber-50 text-amber-600 rounded-xl">
                    <Award className="w-5 h-5" />
                  </div>
                  <div>
                    <h3 className="text-lg font-bold text-slate-900">Highest Run Scorers By Team</h3>
                    <p className="text-xs text-slate-500">Leading run-getters for each franchise</p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {Object.entries(highestRunScorer.highest_run_scorer_for_each_team).map(([team, info]: any) => (
                  <div key={team} className="p-3.5 rounded-xl bg-slate-50 border border-slate-100">
                    <div className="flex justify-between items-start">
                      <span className="px-2 py-0.5 bg-slate-200 text-slate-700 font-bold rounded text-xs">{team}</span>
                      <span className="text-xs text-emerald-600 font-bold">{info.runs} runs</span>
                    </div>
                    <p className="font-semibold text-slate-800 text-sm mt-2">{info.player}</p>
                    <p className="text-[11px] text-slate-400">{info.matches} matches played</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: HEAD-TO-HEAD MATRIX & EXPLORER */}
      {activeTab === 'headToHead' && (
        <div className="space-y-8 animate-fadeIn">
          {/* Interactive Matchup Tool */}
          <div className="bg-gradient-to-br from-slate-900 via-slate-800 to-indigo-950 rounded-2xl p-6 text-white shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <Swords className="w-6 h-6 text-indigo-400" />
              <h2 className="text-xl font-bold">Interactive Matchup Explorer</h2>
            </div>

            {/* Selectors */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
              <div>
                <label className="block text-xs text-slate-400 uppercase font-semibold mb-2">Select Team A</label>
                <select 
                  value={team1} 
                  onChange={(e) => setTeam1(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 text-white rounded-xl p-3 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
                >
                  {teamList.map((t) => (
                    <option key={t} value={t} disabled={t === team2}>{t}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs text-slate-400 uppercase font-semibold mb-2">Select Team B</label>
                <select 
                  value={team2} 
                  onChange={(e) => setTeam2(e.target.value)}
                  className="w-full bg-slate-800 border border-slate-700 text-white rounded-xl p-3 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-semibold"
                >
                  {teamList.map((t) => (
                    <option key={t} value={t} disabled={t === team1}>{t}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Matchup Results Dashboard */}
            {h2hRecord ? (
              <div className="mt-8 pt-8 border-t border-slate-700/60 grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Wins Visual Progress */}
                <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700/50 flex flex-col justify-between">
                  <div>
                    <span className="text-xs text-slate-400 uppercase font-semibold">Total Matches Played</span>
                    <h3 className="text-4xl font-extrabold text-white mt-1">{h2hRecord.matchesPlayed}</h3>
                  </div>

                  <div className="mt-6">
                    <div className="flex justify-between text-xs font-bold mb-2">
                      <span className="text-blue-400">{h2hRecord.team1}: {h2hRecord.team1Wins} Wins</span>
                      <span className="text-indigo-400">{h2hRecord.team2}: {h2hRecord.team2Wins} Wins</span>
                    </div>
                    <div className="w-full h-3 bg-slate-700 rounded-full overflow-hidden flex">
                      <div 
                        className="bg-blue-500 h-full transition-all duration-500" 
                        style={{ width: `${(h2hRecord.team1Wins / h2hRecord.matchesPlayed) * 100}%` }}
                      />
                      <div 
                        className="bg-indigo-500 h-full transition-all duration-500" 
                        style={{ width: `${(h2hRecord.team2Wins / h2hRecord.matchesPlayed) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Team 1 Stats vs Team 2 */}
                <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700/50 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-blue-400">{team1} Stats vs {team2}</span>
                    <span className="text-[10px] bg-blue-500/20 text-blue-300 px-2 py-0.5 rounded">Team Records</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 pt-2">
                    <div className="bg-slate-900/60 p-2.5 rounded-lg text-center">
                      <p className="text-[10px] text-slate-400">Highest Total</p>
                      <p className="text-lg font-black text-emerald-400">{t1VsT2Scores?.highest || 'N/A'}</p>
                    </div>
                    <div className="bg-slate-900/60 p-2.5 rounded-lg text-center">
                      <p className="text-[10px] text-slate-400">Lowest Total</p>
                      <p className="text-lg font-black text-rose-400">{t1VsT2Scores?.lowest || 'N/A'}</p>
                    </div>
                  </div>
                  {t1PlayerVsT2 && (
                    <div className="bg-slate-900/60 p-3 rounded-lg flex items-center justify-between text-xs">
                      <div>
                        <p className="text-slate-400 text-[10px]">Top Score ({team1})</p>
                        <p className="font-bold text-slate-200">{t1PlayerVsT2.player}</p>
                      </div>
                      <span className="font-black text-amber-400 text-sm">{t1PlayerVsT2.score} ({t1PlayerVsT2.year})</span>
                    </div>
                  )}
                </div>

                {/* Team 2 Stats vs Team 1 */}
                <div className="bg-slate-800/80 p-5 rounded-xl border border-slate-700/50 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-indigo-400">{team2} Stats vs {team1}</span>
                    <span className="text-[10px] bg-indigo-500/20 text-indigo-300 px-2 py-0.5 rounded">Team Records</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 pt-2">
                    <div className="bg-slate-900/60 p-2.5 rounded-lg text-center">
                      <p className="text-[10px] text-slate-400">Highest Total</p>
                      <p className="text-lg font-black text-emerald-400">{t2VsT1Scores?.highest || 'N/A'}</p>
                    </div>
                    <div className="bg-slate-900/60 p-2.5 rounded-lg text-center">
                      <p className="text-[10px] text-slate-400">Lowest Total</p>
                      <p className="text-lg font-black text-rose-400">{t2VsT1Scores?.lowest || 'N/A'}</p>
                    </div>
                  </div>
                  {t2PlayerVsT1 && (
                    <div className="bg-slate-900/60 p-3 rounded-lg flex items-center justify-between text-xs">
                      <div>
                        <p className="text-slate-400 text-[10px]">Top Score ({team2})</p>
                        <p className="font-bold text-slate-200">{t2PlayerVsT1.player}</p>
                      </div>
                      <span className="font-black text-amber-400 text-sm">{t2PlayerVsT1.score} ({t2PlayerVsT1.year})</span>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <p className="mt-6 text-sm text-slate-400">No direct head-to-head records found for this team combination.</p>
            )}
          </div>

          {/* Full Head to Head Summary List */}
          <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100">
            <h3 className="text-lg font-bold text-slate-900 mb-4">Complete IPL Rivalry Summary Table</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-600">
                <thead className="bg-slate-50 text-slate-700 uppercase text-[11px] tracking-wider border-b">
                  <tr>
                    <th className="p-3.5">Rivalry</th>
                    <th className="p-3.5">Matches</th>
                    <th className="p-3.5">Team 1 Wins</th>
                    <th className="p-3.5">Team 2 Wins</th>
                    <th className="p-3.5">Win Dominance</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {headToHead.map((row: any) => {
                    const t1Pct = Math.round((row.team1Wins / row.matchesPlayed) * 100);
                    return (
                      <tr key={row._id} className="hover:bg-slate-50/80 transition-colors">
                        <td className="p-3.5 font-bold text-slate-800">{row.team1} vs {row.team2}</td>
                        <td className="p-3.5">{row.matchesPlayed}</td>
                        <td className="p-3.5 font-semibold text-blue-600">{row.team1}: {row.team1Wins}</td>
                        <td className="p-3.5 font-semibold text-indigo-600">{row.team2}: {row.team2Wins}</td>
                        <td className="p-3.5">
                          <div className="w-32 bg-slate-200 h-2 rounded-full overflow-hidden flex">
                            <div className="bg-blue-500 h-full" style={{ width: `${t1Pct}%` }} />
                            <div className="bg-indigo-500 h-full" style={{ width: `${100 - t1Pct}%` }} />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 3: FRANCHISE HISTORY */}
      {activeTab === 'franchises' && (
        <div className="space-y-8 animate-fadeIn">
          <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100">
            <div className="flex items-center gap-3 mb-6">
              <Trophy className="w-6 h-6 text-amber-500" />
              <h2 className="text-xl font-bold text-slate-900">Franchise Win % & Titles Leaderboard</h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {winPercentageHistory.map((teamData: any, idx: number) => (
                <div key={idx} className="p-5 rounded-2xl bg-slate-50 border border-slate-100 hover:border-indigo-200 transition-all shadow-sm">
                  <div className="flex justify-between items-start mb-3">
                    <div>
                      <h3 className="font-bold text-slate-900 text-lg">{teamData.team}</h3>
                      <p className="text-xs text-slate-500">{teamData.total_matches} Total Matches Played</p>
                    </div>
                    <span className="flex items-center gap-1 bg-amber-100 text-amber-800 px-3 py-1 rounded-full font-bold text-xs">
                      <Trophy className="w-3.5 h-3.5" /> {teamData.titles} Titles
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 bg-white p-3 rounded-xl text-center border border-slate-100 my-4">
                    <div>
                      <p className="text-[10px] text-slate-400 uppercase font-semibold">Total Wins</p>
                      <p className="text-base font-black text-emerald-600">{teamData.total_wins}</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-slate-400 uppercase font-semibold">Win Rate</p>
                      <p className="text-base font-black text-indigo-600">{teamData.win_percentage}%</p>
                    </div>
                    <div>
                      <p className="text-[10px] text-slate-400 uppercase font-semibold">Playoffs</p>
                      <p className="text-base font-black text-slate-700">{teamData.times_qualified_for_playoffs} times</p>
                    </div>
                  </div>

                  {/* Season Breakdown Snapshot */}
                  <div>
                    <p className="text-xs font-semibold text-slate-600 mb-2">Recent Season Performances:</p>
                    <div className="flex flex-wrap gap-1.5">
                      {teamData.seasons.slice(-5).map((s: any) => (
                        <span key={s.year} className="text-[11px] bg-slate-200/80 px-2 py-1 rounded font-medium text-slate-700">
                          {s.year}: <strong className="text-slate-900">{s.position}</strong>
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* TAB 4: ALL-TIME RECORDS */}
      {activeTab === 'records' && (
        <div className="space-y-8 animate-fadeIn">
          {/* Global Team Extremes Table */}
          <div className="bg-white rounded-2xl p-6 shadow-md border border-slate-100">
            <h3 className="text-lg font-bold text-slate-900 mb-4">All-Time Team Highest & Lowest Totals</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-slate-600">
                <thead className="bg-slate-50 text-slate-700 uppercase text-[11px] tracking-wider border-b">
                  <tr>
                    <th className="p-3.5">Team</th>
                    <th className="p-3.5 text-emerald-600">Highest Score</th>
                    <th className="p-3.5">Against</th>
                    <th className="p-3.5 text-rose-600">Lowest Score</th>
                    <th className="p-3.5">Against</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {highestLowest.map((row: any) => (
                    <tr key={row.team} className="hover:bg-slate-50/80 transition-colors">
                      <td className="p-3.5 font-bold text-slate-900">{row.team}</td>
                      <td className="p-3.5 font-extrabold text-emerald-600">{row.highest.score}</td>
                      <td className="p-3.5 font-medium">{row.highest.opposition}</td>
                      <td className="p-3.5 font-extrabold text-rose-600">{row.lowest.score}</td>
                      <td className="p-3.5 font-medium">{row.lowest.opposition}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Statistics;