const BASE_URL = "http://localhost:8000";

export interface BattingScore {
    name: string;
    runs: number;
    balls: number;
    fours: number;
    sixes: number;
    strike_rate: number;
    out: boolean;
    dismissal: string;
}

export interface BowlingScore {
    name: string;
    overs: string;
    runs_conceded: number;
    wickets: number;
    economy: number;
}

export interface Innings {
    inning: number;
    batting_team: string;
    bowling_team: string;
    target: number;
    total: {
        runs: number;
        wickets: number;
        overs: string;
    };
    batting: BattingScore[];
    bowling: BowlingScore[];
}

export interface PredictionResult {
    tournament_context: {
        team_a: string;
        team_b: string;
        venue: string;
        toss_winner: string;
        toss_decision: string;
    };

    innings: Innings[];

    result: string;
    winner: string;
    model_backend: string;
}

interface PredictionRequestBody {
    team_a: string;
    team_b: string;
    venue?: string;
}

const PredictionModel = {
    predict: async (
        teamA: string,
        teamB: string,
        venue?: string
    ): Promise<PredictionResult | null> => {

        if (teamA === teamB)
            return null;

        try {

            const body: PredictionRequestBody = {
                team_a: teamA,
                team_b: teamB
            };

            if (venue)
                body.venue = venue;

            const response = await fetch(
                `${BASE_URL}/predict-1-match`,
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(body)
                }
            );

            if (!response.ok)
                throw new Error("Prediction failed");

            return await response.json();

        } catch (err) {

            console.error(err);
            return null;

        }

    }
};

export default PredictionModel;