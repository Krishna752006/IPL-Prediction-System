import { create } from 'zustand';
import scheduleData from '../data/schedule.json';

export interface MatchPrediction {
  matchNo: string;
  winner: string;
  confidence: number; // Win probability percentage
}

export interface TeamStanding {
  team: string;
  played: number;
  won: number;
  lost: number;
  points: number;
}

interface PredictionState {
  predictions: Record<string, MatchPrediction>;
  isPredicting: boolean;
  predictAllMatches: () => Promise<void>;
  clearPredictions: () => void;
  getPointsTable: () => TeamStanding[];
}

const ALL_TEAMS = ['CSK', 'DC', 'GT', 'KKR', 'LSG', 'MI', 'PBKS', 'RCB', 'RR', 'SRH'];

export const usePredictionStore = create<PredictionState>((set, get) => ({
  predictions: {},
  isPredicting: false,

  predictAllMatches: async () => {
    set({ isPredicting: true });

    // Simulate API batch prediction delay
    await new Promise((resolve) => setTimeout(resolve, 1200));

    const newPredictions: Record<string, MatchPrediction> = {};

    Object.entries(scheduleData).forEach(([matchNo, details]) => {
      const matchString = details[0];
      if (!matchString || !matchString.includes(' vs ')) return;

      const [team1, team2] = matchString.split(' vs ').map((t) => t.trim());
      
      // Predict winner (Replace with backend API fetch if desired)
      const randomValue = Math.random();
      const winner = randomValue > 0.48 ? team1 : team2;
      const confidence = Math.floor(52 + Math.random() * 38);

      newPredictions[matchNo] = {
        matchNo,
        winner,
        confidence,
      };
    });

    set({ predictions: newPredictions, isPredicting: false });
  },

  clearPredictions: () => set({ predictions: {} }),

  getPointsTable: () => {
    const standings: Record<string, TeamStanding> = {};

    ALL_TEAMS.forEach((team) => {
      standings[team] = { team, played: 0, won: 0, lost: 0, points: 0 };
    });

    const predictions = get().predictions;

    Object.entries(scheduleData).forEach(([matchNo, details]) => {
      const matchString = details[0];
      const prediction = predictions[matchNo];

      if (prediction && matchString && matchString.includes(' vs ')) {
        const [team1, team2] = matchString.split(' vs ').map((t) => t.trim());
        const winner = prediction.winner;
        const loser = winner === team1 ? team2 : team1;

        if (standings[winner]) {
          standings[winner].played += 1;
          standings[winner].won += 1;
          standings[winner].points += 2;
        }

        if (standings[loser]) {
          standings[loser].played += 1;
          standings[loser].lost += 1;
        }
      }
    });

    return Object.values(standings).sort((a, b) => {
      if (b.points !== a.points) return b.points - a.points;
      return b.won - a.won;
    });
  },
}));