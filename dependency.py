import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Directory containing the .xlsx files
directory = '/Users/habeeb/Downloads/Git/ML/zayyannu/data'

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

# List of pollutants and weather variables
pollutants = ['Pm2.5', 'Pm10', 'So2', 'Co2', 'Co', 'So4']
weather_vars = ['Wind Speed', 'Tmin', 'Tmax', 'Rain']




import matplotlib.pyplot as plt

# Define rolling window size (e.g., 12 months)
rolling_window_size = 12

# Initialize a dictionary to store rolling correlations
rolling_correlations = {}

# Compute rolling correlations for each DataFrame
for df_name, df in dataframes.items():
    for pollutant in pollutants:
        if pollutant in df.columns:
            for weather_var in weather_vars:
                if weather_var in df.columns:
                    # Compute rolling correlation
                    rolling_corr = df[pollutant].rolling(window=rolling_window_size).corr(df[weather_var])
                    
                    # Store results
                    key = f'{df_name} - {pollutant} vs. {weather_var}'
                    rolling_correlations[key] = rolling_corr




# Plot and save rolling correlations
for key, rolling_corr in rolling_correlations.items():
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot the rolling correlation
    ax.plot(rolling_corr.index, rolling_corr, label='Rolling Correlation', color='blue')
    
    # Set plot title and labels
    ax.set_title(f'Rolling Correlation: {key}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Correlation')
    ax.legend()
    ax.grid(True)
    
    # Save the figure
    plot_filename = os.path.join('Explorations', f'{key.replace(" ", "_").replace("/", "_")}.png')
    plt.savefig(plot_filename)
    
    # Clear the current figure
    plt.close(fig)

print("All plots have been saved.")

