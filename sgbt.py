import pandas as pd
import os
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the .xlsx files
directory = '/Users/habeeb/Downloads/Git/ML/zayyannu/data'
# Directory to save performance metrics and plots
exploration_folder = '/Users/habeeb/Downloads/Git/ML/zayyannu/xgb'

# Create exploration folder if it doesn't exist
os.makedirs(exploration_folder, exist_ok=True)

# Dictionary to store the DataFrames with their modified names
dataframes = {}

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith('.xlsx'):
        # Read the file into a DataFrame
        df = pd.read_excel(os.path.join(directory, filename), engine='openpyxl')
        
        # Extract the new name (everything before the underscore)
        new_name = os.path.splitext(filename)[0].split('_')[0]
        
        # Convert 'DATE' column to datetime and set it as index
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        
        # Ensure the data is resampled to monthly frequency
        df = df.resample('M').mean()
        
        # Assign the DataFrame to the dictionary with the new name as the key
        dataframes[new_name] = df

# List of pollutants to predict
pollutants = ['Pm2.5', 'Pm10', 'So2', 'Co2', 'Co', 'So4']

# Initialize a list to store performance metrics
performance_metrics = []

def create_lagged_features(df, lags=1):
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    df = df.dropna()
    return df

# Define hyperparameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

for df_name, df in dataframes.items():
    for pollutant in pollutants:
        if pollutant in df.columns:
            # Prepare the data for XGBoost
            df_xgb = df[[pollutant]].reset_index()
            df_xgb.columns = ['ds', 'y']
            
            # Create lagged features
            df_xgb = create_lagged_features(df_xgb, lags=3)
            
            # Check if there are enough samples
            if df_xgb.empty or len(df_xgb) < 2:
                print(f"Insufficient data for {df_name} - {pollutant}. Skipping...")
                continue
            
            print(f"Processing {df_name} - {pollutant} with {len(df_xgb)} samples.")
            
            # Split data into features and target
            X = df_xgb.drop(columns=['ds', 'y'])
            y = df_xgb['y']
            
            # Normalize features
            scaler = StandardScaler()  # or MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, shuffle=False)
            
            # Check if train or test sets are empty
            if len(X_train) == 0 or len(X_test) == 0 or len(y_train) == 0 or len(y_test) == 0:
                print(f"Split resulted in empty train or test set for {df_name} - {pollutant}.")
                continue
            
            # Initialize XGBoost model
            model = xgb.XGBRegressor(objective='reg:squarederror')
            
            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                scoring='neg_mean_squared_error',
                cv=TimeSeriesSplit(n_splits=5),  # Time series cross-validation
                verbose=1
            )
            
            # Fit GridSearchCV
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate performance metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            # Calculate R-squared
            r2 = best_model.score(X_test, y_test)
            
            # Store the metrics
            performance_metrics.append({
                'DataFrame': df_name,
                'Pollutant': pollutant,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R-squared': r2
            })
            
            # Plot predictions vs actual values
            plt.figure(figsize=(14, 7))
            plt.plot(y_test.index, y_test, label='Actual', color='blue')
            plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')
            plt.title(f'{df_name} - {pollutant} Forecast vs Actual')
            plt.xlabel('Date')
            plt.ylabel(f'{pollutant} Levels')
            plt.legend()
            plt.grid(True)
            
            # Save plot
            plot_path = os.path.join(exploration_folder, f'{df_name}_{pollutant}_forecast_vs_actual.png')
            plt.savefig(plot_path)
            plt.close()

# Convert performance metrics to a DataFrame
performance_df = pd.DataFrame(performance_metrics)

# Save performance metrics to a CSV file
performance_csv_path = os.path.join(exploration_folder, 'xgb_result.csv')
performance_df.to_csv(performance_csv_path, index=False)

print(f"Performance metrics have been saved to {performance_csv_path}")
print(f"Plots have been saved in {exploration_folder}")
