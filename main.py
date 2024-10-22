import numpy as np
import pandas as pd
from cfbd import Configuration, ApiClient, TeamsApi, GamesApi
from sklearn.model_selection import train_test_split

# Configuration
configuration = Configuration()
configuration.api_key['Authorization'] = 'PIX1Z54E93iZiIa+ZPDwFxhiI+pcS+ZsOjQYcPwIUI9dWJDf5/ur3C+X21aXgKVq'  # Replace with your actual API key
configuration.api_key_prefix['Authorization'] = 'Bearer'

api_client = ApiClient(configuration)
teams_api = TeamsApi(api_client)
games_api = GamesApi(api_client)

# Teams and years of interest
teams = [
    "Michigan", "Michigan State", "Ohio State", "Penn State",
    "Maryland", "Indiana", "Rutgers"
]
years = [2023, 2022, 2021, 2020, 2019]

# Fetch team statistics
team_stats_list = []
for year in years:
    for team in teams:
        try:
            response = games_api.get_games(year=year, team=team, season_type='regular')
            games = response  # Adjust based on the actual response structure

            for game in games:
                # Skip if the game hasn't been played yet
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

# Select features and target variables
features = df.drop(columns=required_columns, errors='ignore')
targets = df[['points_for', 'points_against']]

# Normalize features
normalized_features = (features - features.min()) / (features.max() - features.min())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    normalized_features.values, targets.values, test_size=0.2, random_state=42
)

# Initialize weights and biases
input_neurons = X_train.shape[1]
hidden_neurons_1 = 10
hidden_neurons_2 = 5
output_neurons = 2
learning_rate = 0.01
epochs = 10000

np.random.seed(0)
weights_input_hidden1 = np.random.rand(input_neurons, hidden_neurons_1)
weights_hidden1_hidden2 = np.random.rand(hidden_neurons_1, hidden_neurons_2)
weights_hidden2_output = np.random.rand(hidden_neurons_2, output_neurons)


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

    # Calculate error (MSE)
    error = np.mean((y_train - predicted_output) ** 2)

    # Backpropagation
    output_error = y_train - predicted_output
    output_delta = output_error  # Linear derivative is 1

    hidden_layer2_error = output_delta.dot(weights_hidden2_output.T)
    hidden_layer2_delta = hidden_layer2_error * sigmoid_derivative(hidden_layer2_output)

    hidden_layer1_error = hidden_layer2_delta.dot(weights_hidden1_hidden2.T)
    hidden_layer1_delta = hidden_layer1_error * sigmoid_derivative(hidden_layer1_output)

    # Update weights
    weights_hidden2_output += hidden_layer2_output.T.dot(output_delta) * learning_rate
    weights_hidden1_hidden2 += hidden_layer1_output.T.dot(hidden_layer2_delta) * learning_rate
    weights_input_hidden1 += X_train.T.dot(hidden_layer1_delta) * learning_rate

    # Optional: Print error every 1000 epochs
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {error}")

# Testing
hidden_layer1_input = np.dot(X_test, weights_input_hidden1)
hidden_layer1_output = sigmoid(hidden_layer1_input)

hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
hidden_layer2_output = sigmoid(hidden_layer2_input)

predicted_output = np.dot(hidden_layer2_output, weights_hidden2_output)

# Evaluate the model
from sklearn.metrics import mean_squared_error

test_error = mean_squared_error(y_test, predicted_output)
print(f"Test Error: {test_error}")

# Compare predictions with actual values
print("Predictions vs Actual:")
for i in range(len(y_test)):
    print(f"Predicted: {predicted_output[i]}, Actual: {y_test[i]}")