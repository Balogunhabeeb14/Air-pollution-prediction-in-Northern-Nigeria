import pandas as pd
import os
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Directory containing the .xlsx files
directory = '/Users/habeeb/Downloads/Git/ML/zayyannu/data'
# Directory to save charts
exploration_folder = '/Users/habeeb/Downloads/Git/ML/zayyannu/profet'

# Create exploration folder if it doesn't exist
os.makedirs('profet', exist_ok=True)

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

# Initialize a dictionary to store performance metrics
performance_metrics = []

# Create predictions and save plots
for df_name, df in dataframes.items():
    for pollutant in pollutants:
        if pollutant in df.columns:
            # Prepare the data for Prophet
            df_prophet = df[[pollutant]].reset_index()
            df_prophet.columns = ['ds', 'y']
            
            # Drop rows with NaN values
            df_prophet = df_prophet.dropna()
            
            # Check the number of rows after dropping NaNs
            if len(df_prophet) < 2:
                print(f"Not enough data to fit Prophet for {df_name} - {pollutant}")
                continue
            
            # Initialize the Prophet model
            model = Prophet()
            
            # Fit the model
            model.fit(df_prophet)
            
            # Create a DataFrame with future dates for prediction
            future = model.make_future_dataframe(periods=12, freq='M')  # Predict 12 months into the future
            forecast = model.predict(future)
            
            # Merge historical data with forecast data
            df_merged = df_prophet.merge(forecast[['ds', 'yhat']], on='ds', how='left')
            
            # Calculate performance metrics
            y_true = df_merged['y']
            y_pred = df_merged['yhat']
            
            # Remove NaNs that might be present after merging
            valid_indices = ~y_true.isna() & ~y_pred.isna()
            y_true = y_true[valid_indices]
            y_pred = y_pred[valid_indices]
            
            if len(y_true) < 2:
                print(f"Not enough valid data to calculate metrics for {df_name} - {pollutant}")
                continue
            
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            
            # Calculate R-squared
            ss_total = np.sum((y_true - np.mean(y_true))**2)
            ss_res = np.sum((y_true - y_pred)**2)
            r2 = 1 - (ss_res / ss_total)
            
            # Store the metrics
            performance_metrics.append({
                'DataFrame': df_name,
                'Pollutant': pollutant,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R-squared': r2
            })
            
            # Plot the forecast
            fig = model.plot(forecast)
            plt.title(f'Pollutant Forecast: {df_name} - {pollutant}')
            
            # Save the plot
            plot_filename = os.path.join('profet', f'{df_name}_{pollutant}_forecast.png')
            plt.savefig(plot_filename)
            plt.close(fig)

# Convert performance metrics to a DataFrame
performance_df = pd.DataFrame(performance_metrics)
print(performance_df)

performance_df.to_csv('profet_result.csv')
