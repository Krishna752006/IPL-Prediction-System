import React, { useState } from "react";
import { Brain, Loader2 } from "lucide-react";
import PredictionModel, {
  PredictionResult,
} from "../utils/PredictionModel";
import { savePredictionHistory } from "../api/historyApi";
import { useAuthStore } from "../store/authStore";

const Predictions: React.FC = () => {
  const { user } = useAuthStore();
  const [team1, setTeam1] = useState("");
  const [team2, setTeam2] = useState("");
  const [venue, setVenue] = useState("");

  const [prediction, setPrediction] =
    useState<PredictionResult | null>(null);

  const [loading, setLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  const teams = [
    { code: "CSK", name: "Chennai Super Kings" },
    { code: "MI", name: "Mumbai Indians" },
    { code: "RCB", name: "Royal Challengers Bengaluru" },
    { code: "KKR", name: "Kolkata Knight Riders" },
    { code: "SRH", name: "Sunrisers Hyderabad" },
    { code: "RR", name: "Rajasthan Royals" },
    { code: "PBKS", name: "Punjab Kings" },
    { code: "DC", name: "Delhi Capitals" },
    { code: "GT", name: "Gujarat Titans" },
    { code: "LSG", name: "Lucknow Super Giants" },
  ];

  const venues = [
    "ACA-VDCA Stadium, Visakhapatnam",
    "Arun Jaitley Stadium, Delhi",
    "Barsapara Stadium, Guwahati",
    "Chepauk Stadium, Chennai",
    "Chinnaswamy Stadium, Bengaluru",
    "DY Patil Stadium, Mumbai",
    "Eden Gardens, Kolkata",
    "Ekana Stadium, Lucknow",
    "Green Park, Kanpur",
    "HPCA Stadium, Dharamsala",
    "Holkar Stadium, Indore",
    "IS Bindra Stadium, Mohali",
    "JSCA Stadium, Ranchi",
    "MCA Stadium, Pune",
    "Mullanpur Stadium, Chandigarh",
    "Narendra Modi Stadium, Ahmedabad",
    "Rajiv Gandhi Stadium, Hyderabad",
    "SCA Stadium, Rajkot",
    "SMS Stadium, Jaipur",
    "SVNS Stadium, Raipur",
    "Sahara Stadium, Pune",
    "VCA Stadium, Nagpur",
    "Wankhede Stadium, Mumbai",
];

  const handlePredict = async () => {
    if (!team1 || !team2) {
      setErrorMessage("Please select both teams.");
      return;
    }

    if (team1 === team2) {
      setErrorMessage("Please select two different teams.");
      return;
    }

    setLoading(true);
    setErrorMessage("");

    const result = await PredictionModel.predict(
      team1,
      team2,
      venue || undefined
    );

    if (result) {

      setPrediction(result);

      const history = {

          team_a: result.tournament_context.team_a,

          team_b: result.tournament_context.team_b,

          venue: result.tournament_context.venue,

          winner: result.winner,

          result: result.result

      };

      if (user?.isGuest) {

          const existing = JSON.parse(
              localStorage.getItem("predictionHistory") || "[]"
          );

          existing.unshift({

              ...history,

              predicted_at: new Date().toISOString()

          });

          localStorage.setItem(
              "predictionHistory",
              JSON.stringify(existing)
          );

      } else {

          await savePredictionHistory(history);

      }

  }

    setLoading(false);
  };

  const getTeamName = (code: string) =>
    teams.find((t) => t.code === code)?.name || code;

  return (
    <div className="max-w-5xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">

        <div className="flex justify-center mb-6">
          <Brain className="h-12 w-12 text-indigo-600" />
        </div>

        <h2 className="text-3xl font-bold text-center mb-8">
          IPL Match Prediction
        </h2>

        <div className="grid md:grid-cols-3 gap-4">

          <div>
            <label className="block text-sm font-medium mb-2">
              Team 1
            </label>

            <select
              value={team1}
              onChange={(e) => setTeam1(e.target.value)}
              className="w-full border rounded-md p-2"
            >
              <option value="">Select Team</option>

              {teams.map((team) => (
                <option key={team.code} value={team.code}>
                  {team.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Team 2
            </label>

            <select
              value={team2}
              onChange={(e) => setTeam2(e.target.value)}
              className="w-full border rounded-md p-2"
            >
              <option value="">Select Team</option>

              {teams.map((team) => (
                <option key={team.code} value={team.code}>
                  {team.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium mb-2">
              Venue (Optional)
            </label>

            <select
              value={venue}
              onChange={(e) => setVenue(e.target.value)}
              className="w-full border rounded-md p-2"
            >
              <option value="">Random Venue</option>

              {venues.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>

        </div>

        <button
          onClick={handlePredict}
          disabled={loading}
          className="mt-6 w-full bg-indigo-600 hover:bg-indigo-700 text-white rounded-md py-3 transition"
        >
          {loading ? (
            <div className="flex justify-center items-center gap-2">
              <Loader2 className="animate-spin h-5 w-5" />
              Predicting...
            </div>
          ) : (
            "Predict Match"
          )}
        </button>

        {errorMessage && (
          <div className="mt-4 text-red-600 font-medium">
            {errorMessage}
          </div>
        )}

        {/* Prediction Result starts here */}
                {prediction && (
          <div className="mt-8 space-y-6">

            {/* Match Summary */}
            <div className="bg-indigo-50 rounded-lg p-6 border border-indigo-200">
              <h3 className="text-2xl font-bold mb-4">
                Match Summary
              </h3>

              <div className="grid md:grid-cols-2 gap-4">

                <div>
                  <p>
                    <strong>Team A:</strong>{" "}
                    {getTeamName(
                      prediction.tournament_context.team_a
                    )}
                  </p>

                  <p>
                    <strong>Team B:</strong>{" "}
                    {getTeamName(
                      prediction.tournament_context.team_b
                    )}
                  </p>

                  <p>
                    <strong>Venue:</strong>{" "}
                    {prediction.tournament_context.venue}
                  </p>
                </div>

                <div>
                  <p>
                    <strong>Toss Winner:</strong>{" "}
                    {getTeamName(
                      prediction.tournament_context.toss_winner
                    )}
                  </p>

                  <p>
                    <strong>Toss Decision:</strong>{" "}
                    {prediction.tournament_context.toss_decision}
                  </p>

                  <p>
                    <strong>Winner:</strong>{" "}
                    <span className="text-green-700 font-bold">
                      {getTeamName(prediction.winner)}
                    </span>
                  </p>

                  <p>
                    <strong>Result:</strong>{" "}
                    {prediction.result}
                  </p>
                </div>

              </div>
            </div>

            {/* Innings */}

            {prediction.innings.map((inning) => (
              <div
                key={inning.inning}
                className="bg-white border rounded-lg shadow-md p-6"
              >
                <h3 className="text-xl font-bold mb-2">
                  Innings {inning.inning}
                </h3>

                <p className="text-lg font-semibold mb-1">
                  {getTeamName(inning.batting_team)}
                </p>

                <p className="mb-5 text-gray-700">
                  {inning.total.runs}/{inning.total.wickets} (
                  {inning.total.overs} overs)
                </p>

                {/* Batting */}

                <h4 className="text-lg font-semibold mb-2">
                  Batting Scorecard
                </h4>

                <div className="overflow-x-auto">

                  <table className="min-w-full border border-gray-300 text-sm">

                    <thead className="bg-gray-100">
                      <tr>
                        <th className="border px-3 py-2 text-left">
                          Batter
                        </th>

                        <th className="border px-3 py-2">
                          R
                        </th>

                        <th className="border px-3 py-2">
                          B
                        </th>

                        <th className="border px-3 py-2">
                          4s
                        </th>

                        <th className="border px-3 py-2">
                          6s
                        </th>

                        <th className="border px-3 py-2">
                          SR
                        </th>

                        <th className="border px-3 py-2">
                          Dismissal
                        </th>
                      </tr>
                    </thead>

                    <tbody>

                      {inning.batting.map((player) => (

                        <tr key={player.name}>

                          <td className="border px-3 py-2">
                            {player.name}
                          </td>

                          <td className="border px-3 py-2 text-center">
                            {player.runs}
                          </td>

                          <td className="border px-3 py-2 text-center">
                            {player.balls}
                          </td>

                          <td className="border px-3 py-2 text-center">
                            {player.fours}
                          </td>

                          <td className="border px-3 py-2 text-center">
                            {player.sixes}
                          </td>

                          <td className="border px-3 py-2 text-center">
                            {player.strike_rate.toFixed(2)}
                          </td>

                          <td className="border px-3 py-2">
                            {player.out
                              ? player.dismissal
                              : "Not Out"}
                          </td>

                        </tr>

                      ))}

                    </tbody>

                  </table>

                </div>

                {/* Bowling */}

                <h4 className="text-lg font-semibold mt-8 mb-2">
                  Bowling Figures
                </h4>

                <div className="overflow-x-auto">

                  <table className="min-w-full border border-gray-300 text-sm">

                    <thead className="bg-gray-100">
                      <tr>

                        <th className="border px-3 py-2 text-left">
                          Bowler
                        </th>

                        <th className="border px-3 py-2">
                          Overs
                        </th>

                        <th className="border px-3 py-2">
                          Runs
                        </th>

                        <th className="border px-3 py-2">
                          Wickets
                        </th>

                        <th className="border px-3 py-2">
                          Economy
                        </th>

                      </tr>
                    </thead>

                    <tbody>

                      {inning.bowling.map((bowler) => (

                        <tr key={bowler.name}>

                          <td className="border px-3 py-2">
                            {bowler.name}
                          </td>

                          <td className="border px-3 py-2 text-center">
                            {bowler.overs}
                          </td>

                          <td className="border px-3 py-2 text-center">
                            {bowler.runs_conceded}
                          </td>

                          <td className="border px-3 py-2 text-center">
                            {bowler.wickets}
                          </td>

                          <td className="border px-3 py-2 text-center">
                            {bowler.economy.toFixed(2)}
                          </td>

                        </tr>

                      ))}

                    </tbody>

                  </table>

                </div>

              </div>
            ))}

          </div>
        )}

      </div>
    </div>
  );
};

export default Predictions;