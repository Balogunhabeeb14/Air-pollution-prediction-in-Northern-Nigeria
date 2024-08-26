import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version: ", tf.__version__)

# Directory containing the .xlsx files
directory = '/Users/habeeb/Downloads/Git/ML/zayyannu/data'
# Directory to save performance metrics and plots
exploration_folder = '/Users/habeeb/Downloads/Git/ML/zayyannu/dl'

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

# Standardize column names
def standardize_column_names(df):
    columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df.columns = columns
    return df

# Apply standardization to all dataframes
for key in dataframes:
    dataframes[key] = standardize_column_names(dataframes[key])


# List of pollutants to check
pollutants = ['co2']

# Handle missing pollutants
for key, df in dataframes.items():
    for pollutant in pollutants:
        if pollutant not in df.columns:
            print(f"Warning: {pollutant} missing in {key}. Adding column with NaN values.")
            df[pollutant] = np.nan
    df.fillna(method='ffill', inplace=True)
    df.fillna(0, inplace=True)



# Scaling the data
scalers = {}
scaled_data = {}

for key, df in dataframes.items():
    if df.empty:
        print(f"Warning: DataFrame for {key} is empty.")
        continue
    
    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)
    scalers[key] = scaler
    scaled_data[key] = scaled_df


# Print scaler information
for key, scaler in scalers.items():
    print(f"Scaler for {key}: {scaler}")
    
# Function to create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Set sequence length (number of past months to use for prediction)
seq_length = 12


# Create sequences and split into training and testing sets
X_train, y_train = {}, {}
X_test, y_test = {}, {}

for key, data in scaled_data.items():
    X, y = create_sequences(data, seq_length)
    # Split data into training and test sets (e.g., last 12 months for testing)
    train_size = len(X) - 12
    if train_size > 0:
        X_train[key], y_train[key] = X[:train_size], y[:train_size]
        X_test[key], y_test[key] = X[train_size:], y[train_size:]
    else:
        print(f"Not enough data for {key}")


# Check sequence shapes
for key, (X, y) in zip(X_train.keys(), zip(X_train.values(), y_train.values())):
    print(f"Sequences for {key} - X shape: {X.shape}, y shape: {y.shape}")


# Define LSTM model
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Train the model for each pollutant
models = {}

for pollutant in pollutants:
    if pollutant in X_train:
        input_shape = (X_train[pollutant].shape[1], X_train[pollutant].shape[2])
        model = create_lstm_model(input_shape)
        
        # Early stopping to avoid overfitting
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train the model
        with tf.device('/GPU:0'):  # Ensures training uses the GPU
            model.fit(X_train[pollutant], y_train[pollutant], validation_split=0.2, epochs=100, batch_size=16, callbacks=[es])
        
        models[pollutant] = model
    else:
        print(f"Skipping {pollutant} as it's missing in X_train")


# Model summaries
for pollutant, model in models.items():
    print(f"Model summary for {pollutant}:")
    model.summary()

# Save models if needed
for pollutant, model in models.items():
    model.save(f"{exploration_folder}/{pollutant}_lstm_model.h5")

predictions = {}
for pollutant in pollutants:
    if pollutant in models:
        # Make predictions
        predicted = models[pollutant].predict(X_test[pollutant])
        
        # Inverse transform to get actual values
        predicted = scalers[pollutant].inverse_transform(predicted)
        y_true = scalers[pollutant].inverse_transform(y_test[pollutant])
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, predicted))
        print(f"RMSE for {pollutant}: {rmse}")
        
        predictions[pollutant] = predicted


# Sample predictions
for pollutant in predictions:
    print(f"Sample predictions for {pollutant}: {predictions[pollutant][:5]}")
    print(f"Sample true values for {pollutant}: {y_test[pollutant][:5]}")
    
    
# Plot predictions vs actual values
for pollutant in predictions:
    if pollutant in X_train:
        plt.figure(figsize=(10, 6))
        plt.plot(predictions[pollutant], label='Predicted')
        plt.plot(y_test[pollutant], label='Actual')
        plt.title(f"{pollutant} Prediction")
        plt.legend()
        plt.show()

future_predictions = {}

for pollutant in predictions:
    # Take the last 12 months of the training set
    last_sequence = X_train[pollutant][-1]
    
    # Predict the next 12 months
    future_pred = []
    for _ in range(12):
        pred = models[pollutant].predict(np.expand_dims(last_sequence, axis=0))
        future_pred.append(pred)
        # Update the sequence with the new prediction
        last_sequence = np.vstack([last_sequence[1:], pred])
    
    # Inverse transform
    future_pred = scalers[pollutant].inverse_transform(future_pred)
    future_predictions[pollutant] = future_pred

# Plot future predictions
for pollutant in future_predictions:
    plt.figure(figsize=(10, 6))
    plt.plot(future_predictions[pollutant], label='Future Predictions')
    plt.title(f"Future {pollutant} Predictions")
    plt.legend()
    plt.show()