import numpy as np
import pandas as pd
#import os
from cfbd import Configuration, ApiClient, TeamsApi, GamesApi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error

# Securely load the API key from an environment variable
#api_key = os.getenv('CFBD_API_KEY')
#if not api_key:
    #raise ValueError("API key not found. Please set the CFBD_API_KEY environment variable.")

# Configuration
configuration = Configuration()
#configuration.api_key['Authorization'] = api_key
configuration.api_key[
    'Authorization'] = 'PIX1Z54E93iZiIa+ZPDwFxhiI+pcS+ZsOjQYcPwIUI9dWJDf5/ur3C+X21aXgKVq'
configuration.api_key_prefix['Authorization'] = 'Bearer'

api_client = ApiClient(configuration)
teams_api = TeamsApi(api_client)
games_api = GamesApi(api_client)

# Teams and years of interest
teams = [
    "Michigan", "Michigan State", "Ohio State", "Penn State",
    "Maryland", "Indiana", "Rutgers"
]
years = [2024, 2023, 2022, 2021, 2020, 2019]

# Fetch team statistics
team_stats_list = []
for year in years:
    for team in teams:
        try:
            response = games_api.get_games(year=year, team=team, season_type='regular')
            games = response

            for game in games:
                # Skip if the 2024 game hasn't been played yet
                if game.home_points is None or game.away_points is None:
                    continue

                # Determine if the team is home or away
                if game.home_team == team:
                    opponent = game.away_team
                    points_for = game.home_points
                    points_against = game.away_points
                else:
                    opponent = game.home_team
                    points_for = game.away_points
                    points_against = game.home_points

                stats_dict = {
                    'team': team,
                    'opponent': opponent,
                    'points_for': points_for,
                    'points_against': points_against,
                    'year': year,
                    'week': game.week
                }

                team_stats_list.append(stats_dict)
        except Exception as e:
            print(f"Error fetching data for {team} in {year}: {e}")

# Convert to DataFrame
df = pd.DataFrame(team_stats_list)
print(f"Number of records fetched: {len(df)}")

if df.empty:
    raise ValueError("No data fetched. Please check your API calls and parameters.")

# Preprocess Data
df.fillna(0, inplace=True)

# Check for required columns
required_columns = ['team', 'opponent', 'points_for', 'points_against', 'year', 'week']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
    # Handle missing columns as needed
    for col in missing_columns:
        df[col] = 0  # or any other default value

# Define target variables
targets = df[['points_for', 'points_against']]

# Define feature columns
feature_columns = ['team', 'opponent', 'year', 'week']
features = df[feature_columns]

# Retain 'team' and 'opponent' names for output
team_opponent = df[['team', 'opponent']]

features_encoded = pd.get_dummies(features, columns=['team', 'opponent'], drop_first=True)

# Check if features are present after encoding
if features_encoded.empty:
    raise ValueError("No features available after encoding. Please ensure that feature columns are correctly specified.")

# Standardize features
scaler_features = StandardScaler()
standardized_features = scaler_features.fit_transform(features_encoded)

# Scale target variables from 0 to 1
scaler_targets = MinMaxScaler()
scaled_targets = scaler_targets.fit_transform(targets)

# Split data into training and testing sets
X_train, X_test, y_train, y_test, team_train, team_test = train_test_split(
    standardized_features, scaled_targets, team_opponent, test_size=0.2, random_state=42
)

# Initialize weights and biases using Xavier Initialization
def xavier_init(size_in, size_out):
   
    limit = np.sqrt(6 / (size_in + size_out))
    return np.random.uniform(-limit, limit, (size_in, size_out))

input_neurons = X_train.shape[1]
hidden_neurons_1 = 10
hidden_neurons_2 = 5
output_neurons = 2
learning_rate = 0.01
epochs = 10000

np.random.seed(0)
weights_input_hidden1 = xavier_init(input_neurons, hidden_neurons_1)
weights_hidden1_hidden2 = xavier_init(hidden_neurons_1, hidden_neurons_2)
weights_hidden2_output = xavier_init(hidden_neurons_2, output_neurons)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer1_input = np.dot(X_train, weights_input_hidden1)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    predicted_output = np.dot(hidden_layer2_output, weights_hidden2_output)
    # For regression, using linear activation in the output layer is common.

    # Calculate error (MSE)
    error = np.mean((y_train - predicted_output) ** 2)

    # Backpropagation
    output_error = y_train - predicted_output
    output_delta = output_error  # Derivative of linear activation is 1

    hidden_layer2_error = np.dot(output_delta, weights_hidden2_output.T)
    hidden_layer2_delta = hidden_layer2_error * sigmoid_derivative(hidden_layer2_output)

    hidden_layer1_error = np.dot(hidden_layer2_delta, weights_hidden1_hidden2.T)
    hidden_layer1_delta = hidden_layer1_error * sigmoid_derivative(hidden_layer1_output)

    # Update weights
    weights_hidden2_output += np.dot(hidden_layer2_output.T, output_delta) * learning_rate
    weights_hidden1_hidden2 += np.dot(hidden_layer1_output.T, hidden_layer2_delta) * learning_rate
    weights_input_hidden1 += np.dot(X_train.T, hidden_layer1_delta) * learning_rate

    # Optional: Print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {error}")

# Testing
hidden_layer1_input = np.dot(X_test, weights_input_hidden1)
hidden_layer1_output = sigmoid(hidden_layer1_input)

hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
hidden_layer2_output = sigmoid(hidden_layer2_input)

predicted_output = np.dot(hidden_layer2_output, weights_hidden2_output)

# Inverse transform the predictions and actual values
predicted_output_unscaled = scaler_targets.inverse_transform(predicted_output)
y_test_unscaled = scaler_targets.inverse_transform(y_test)

# Evaluate the model
test_error = mean_squared_error(y_test_unscaled, predicted_output_unscaled)
print(f"\nTest MSE: {test_error}\n")

# Compare predictions with actual values along with team names
print("Predictions vs Actual:")
print("Team vs Opponent | Team Predicted | Opponent Predicted | Team Actual | Opponent Actual")
print("-" * 80)
for i in range(min(len(y_test_unscaled), 10)):  # Limiting to first 10 for readability
    team = team_test.iloc[i]['team']
    opponent = team_test.iloc[i]['opponent']
    pred_for = predicted_output_unscaled[i][0]
    pred_against = predicted_output_unscaled[i][1]
    actual_for = y_test_unscaled[i][0]
    actual_against = y_test_unscaled[i][1]
    print(f"{team} vs {opponent} | {pred_for:.2f} | {pred_against:.2f} | {actual_for} | {actual_against}")
