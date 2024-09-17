import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Directory containing the .xlsx files
directory = '/Users/habeeb/Downloads/Git/ML/Air-pollution-prediction-in-Northern-Nigeria/data'
# Directory to save charts
exploration_folder = '/Users/habeeb/Downloads/Git/ML/Air-pollution-prediction-in-Northern-Nigeria/lstm'

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

# Initialize a dictionary to store performance metrics
performance_metrics = []

# Helper function to create LSTM dataset
def create_lstm_dataset(data, time_steps=1):
    X, Y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps), :])
        Y.append(data[i + time_steps, 0])
    return np.array(X), np.array(Y)

# Create predictions and save plots
for df_name, df in dataframes.items():
    for pollutant in pollutants:
        if pollutant in df.columns:
            # Prepare the data for LSTM
            required_columns = [pollutant, 'Wind Speed', 'Tmin', 'Tmax', 'Rain']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Skipping {pollutant} in {df_name} due to missing columns: {missing_columns}")
                continue
            
            df_lstm = df[required_columns].dropna()
            
            # Check if the DataFrame is empty
            if df_lstm.empty:
                print(f"Data for {pollutant} in {df_name} is empty after filtering.")
                continue

            # Print the shape of df_lstm
            print(df_lstm.head())
            print(df_lstm.shape)
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_scaled = scaler.fit_transform(df_lstm)
            
            # Define time steps (you can adjust the window size)
            time_steps = 12
            X, Y = create_lstm_dataset(df_scaled, time_steps)


            # Print the shape of X and Y
            print(f"X shape: {X.shape}, Y shape: {Y.shape}")
            
            
            # Split data into training and testing (80% train, 20% test)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            Y_train, Y_test = Y[:train_size], Y[train_size:]

            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))

            model.add(LSTM(units=25, return_sequences=True))
            model.add(Dropout(0.2))

            #model.add(LSTM(units=32, return_sequences=True))
            #model.add(Dropout(0.2))
            
            model.add(LSTM(units=16, return_sequences=False))
            model.add(Dropout(0.2))


            # Output layer with 1 unit for regression task
            model.add(Dense(units=1))

            # Compile the model
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.summary()

            # Early stopping to avoid overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

            # Train the model
            model.fit(X_train, Y_train, epochs=50, batch_size=8, validation_split=0.2, callbacks=[early_stopping], verbose=1)
            
            # Predict on test data
            Y_pred = model.predict(X_test)

            # Inverse scaling for prediction and true values
            Y_test = scaler.inverse_transform(np.concatenate([Y_test.reshape(-1, 1), X_test[:, -1, 1:]], axis=1))[:, 0]
            Y_pred = scaler.inverse_transform(np.concatenate([Y_pred, X_test[:, -1, 1:]], axis=1))[:, 0]

            # Calculate performance metrics
            mae = mean_absolute_error(Y_test, Y_pred)
            mse = mean_squared_error(Y_test, Y_pred)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((Y_test - Y_pred) / Y_test)) * 100
            
            # Calculate R-squared
            ss_total = np.sum((Y_test - np.mean(Y_test)) ** 2)
            ss_res = np.sum((Y_test - Y_pred) ** 2)
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
            
            # Combine historical and prediction dates
            historical_dates = df.index[-len(Y_test):]
            prediction_dates = df.index[-len(Y_pred):]

            # Plot the historical data and predictions
            plt.figure(figsize=(14, 7))
            plt.plot(df.index, df[pollutant], label='Historical Data', color='blue')
            plt.plot(prediction_dates, Y_pred, label='Predicted Data', color='red')
            plt.title(f'{pollutant} Forecast: {df_name}')
            plt.xlabel('Date')
            plt.ylabel(f'{pollutant} Levels')
            plt.legend()
            
            # Save the plot
            try:
                plot_filename = os.path.join(exploration_folder, f'{df_name}_{pollutant}_lstm_forecast.png')
                plt.savefig(plot_filename)
                print(f"Saved plot for {df_name}_{pollutant} at {plot_filename}")
            except Exception as e:
                print(f"Error saving plot for {df_name}_{pollutant}: {e}")
            plt.close()

# Convert performance metrics to a DataFrame
performance_df = pd.DataFrame(performance_metrics)
print(performance_df)

# Save performance metrics to CSV
performance_df.to_csv(os.path.join(exploration_folder, 'lstm_result.csv'))
