from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import warnings
import numpy as np
import pandas as pd
import math
import threading  
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}}, 
     supports_credentials=True, 
     allow_headers=['Content-Type', 'Authorization', 'Access-Control-Allow-Headers'], 
     methods=['GET', 'POST', 'OPTIONS'])

data = pd.read_csv("data1.csv")
#data.drop(['Unnamed: 0'], axis=1, inplace=True)

input_features = ['batting_team', 'bowling_team', 'venue']
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(data[input_features])

xfeats = ['over', 'ball']
scalerx = StandardScaler()
scaled_x = scalerx.fit_transform(data[xfeats])

yfeats = ['runs', 'wickets']
scalery = StandardScaler()
scaled_y = scalery.fit_transform(data[yfeats])

model = keras.models.load_model("lstm1.keras")

team = {1: 'Chennai Super Kings', 2: 'Delhi Capitals', 3: 'Gujarat Titans', 5: 'Kolkata Knight Riders', 6: 'Lucknow Super Giants', 7: 'Mumbai Indians', 9: 'Punjab Kings', 10: 'Rajasthan Royals', 12: 'Royal Challengers Bengaluru', 13: 'Sunrisers Hyderabad'}
venue = {1: 'Arun Jaitley Stadium', 3: 'Barsapara Cricket Stadium', 7: 'Dr DY Patil Sports Academy', 10: 'Eden Gardens', 11: 'Ekana Cricket Stadium', 12: 'Feroz Shah Kotla', 18: 'M Chinnaswamy Stadium', 19: 'MA Chidambaram Stadium', 20: 'Maharaja Yadavindra Singh International Cricket Stadium', 22: 'Narendra Modi Stadium', 27: 'Punjab Cricket Association IS Bindra Stadium', 28: 'Punjab Cricket Association Stadium', 29: 'Rajiv Gandhi International Stadium', 30: 'Sardar Patel Stadium', 31: 'Saurashtra Cricket Association Stadium', 32: 'Sawai Mansingh Stadium', 35: 'Sheikh Zayed Stadium', 37: 'Subrata Roy Sahara Stadium', 39: 'Vidarbha Cricket Association Stadium', 40: 'Wankhede Stadium'}
steam = {str(key): value for key, value in team.items()}
svenue = {str(key): value for key, value in venue.items()}

def train_model_with_new_data(batfirst, batsecond):
    """Function to train the model asynchronously."""
    try:
        global model  # Ensure we are modifying the global model
        print("Started background training...")

        df = pd.concat([data, batfirst, batsecond], ignore_index=True)
        df.to_csv('data1.csv', index=False)

        encoded_categorical = encoder.fit_transform(df[input_features])
        scaled_x = scalerx.fit_transform(df[xfeats])
        scaled_y = scalery.fit_transform(df[yfeats])

        X = np.hstack((encoded_categorical, scaled_x))
        X = X.reshape(X.shape[0], 1, X.shape[1])
        y = np.vstack((scaled_y))

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
            keras.layers.LSTM(64, return_sequences=True),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(2)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        model.fit(X_train, y_train, epochs=2, batch_size=16, validation_data=(X_val, y_val), verbose=0)
        model.save("lstm1.keras")
    except Exception as e:
        print(f"Error during background training: {e}")

@app.route('/api/predict', methods=['POST'])
def predict_match():
    data = request.json

    if not data:
        return jsonify({'error': 'No data provided.'}), 400
    
    try:
        teamA = int(data.get('teamA'))
        teamB = int(data.get('teamB'))
        venue1 = int(data.get('venue'))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid input format.'}), 400

    if teamA == teamB:
        return jsonify({'error': 'Please provide two different teams as input.'}), 400

    if str(teamA) not in steam or str(teamB) not in steam or str(venue1) not in svenue:
        return jsonify({'error': 'Invalid team or venue code.'}), 400

    try:
        Input1 = [[teamA, teamB, venue1]]
        Input2 = [[teamB, teamA, venue1]]

        input_columns = ['batting_team', 'bowling_team', 'venue']
        Input1_df = pd.DataFrame(Input1, columns=input_columns)
        Input2_df = pd.DataFrame(Input2, columns=input_columns)

        scaled_input1 = encoder.transform(Input1_df)
        scaled_input2 = encoder.transform(Input2_df)

        Extrainput = [[19, 6]]
        scaled_extrainput = scalerx.transform(Extrainput)

        X_give1 = np.hstack((scaled_input1, scaled_extrainput))
        X_give1 = X_give1.reshape(X_give1.shape[0], 1, X_give1.shape[1])

        X_give2 = np.hstack((scaled_input2, scaled_extrainput))
        X_give2 = X_give2.reshape(X_give2.shape[0], 1, X_give2.shape[1])
    except Exception as e:
        return jsonify({'error': f'Error during input preparation: {str(e)}'}), 400

    try:
        pred1 = model.predict(X_give1, verbose=0)
        pred1 = scalery.inverse_transform(pred1).astype(float) 

        pred2 = model.predict(X_give2, verbose=0)
        pred2 = scalery.inverse_transform(pred2).astype(float) 
    except Exception as e:
        return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

    result = {
        'teamA': {
            'score': round(float(pred1[0, 0])), 
            'wickets': round(float(pred1[0, 1])), 
            'run_rate': round(float(pred1[0, 0]) / 20, 2) 
        },
        'teamB': {
            'score': round(float(pred2[0, 0])),  
            'wickets': round(float(pred2[0, 1])), 
            'run_rate': round(float(pred2[0, 0]) / 20, 2)  
        }
    }

    for key1, value1 in team.items():
        if teamA == key1:
            team1_score = round(pred1[0, 0])
            team1_wickets = round(pred1[0, 1])
            team1_run_rate = team1_score / 20
            break

    for key2, value2 in team.items():
        if teamB == key2:
            team2_score = round(pred2[0, 0])
            team2_wickets = round(pred2[0, 1])
            team2_run_rate = team2_score / 20
        
            ballsfaced = int((team1_score / team2_run_rate) * 6)
            oversface = ballsfaced // 6
            remaining_balls = ballsfaced % 6

            if team2_score < team1_score:
                result['teamB']['match_result'] = f"{steam[str(teamA)]} won by {round(float(pred1[0, 0] - float(pred2[0, 0])))} runs"
                winner = key1  
                team2_scored = team2_score
            else:
                team2_scored = math.ceil(round(team1_score / team2_run_rate, 1) * team2_run_rate)
                if team1_score == team2_scored and team2_run_rate > team1_run_rate:
                    team2_scored += 1 
                result['teamB']['score'] = team2_scored
                result['teamB']['match_result'] = f"{value2} chased down the score in {oversface}.{remaining_balls} overs"
                winner = key2  

    batfirst = pd.DataFrame({
    'batting_team': [teamA],  
    'bowling_team': [teamB],
    'over': [19],  
    'ball': [6],   
    'venue': [venue1],
    'runs': [team1_score],
    'wickets': [team1_wickets],
    'winner': [winner]  
    })

    batsecond = pd.DataFrame({
    'batting_team': [teamB],
    'bowling_team': [teamA],
    'over': [oversface],  
    'ball': [remaining_balls],  
    'venue': [venue1],
    'runs': [team2_scored], 
    'wickets': [team2_wickets],
    'winner': [winner] 
    })

    threading.Thread(target=train_model_with_new_data, args=(batfirst, batsecond)).start()
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5173)
