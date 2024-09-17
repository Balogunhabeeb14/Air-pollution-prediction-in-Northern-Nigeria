import pandas as pd
import matplotlib.pyplot as plt
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
        
        # Assign the DataFrame to the dictionary with the new name as the key
        dataframes[new_name] = df

# List all DataFrames
for df_name, df in dataframes.items():
    print(f"DataFrame: {df_name}")
    print(df.head())  # Display the first few rows of each DataFrame
    print("\n")


# List of pollutants to visualize
pollutants = ['Pm2.5', 'Pm10', 'So2', 'Co2', 'Co', 'So4']

# Initialize a dictionary to store the figures for each pollutant
figures = {}

for pollutant in pollutants:
    # Initialize the plot for the current pollutant
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Iterate through each DataFrame and plot the selected pollutant
    for df_name, df in dataframes.items():
        # Check if the required columns exist in the DataFrame
        if pollutant in df.columns:
            ax.plot(df['DATE'], df[pollutant], label=df_name)
    
    # Set plot title and labels
    ax.set_title(f'{pollutant} Levels Over Time Across All Northern Nigeria States')
    ax.set_xlabel('Date')
    ax.set_ylabel(f'{pollutant} Values')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    ax.grid(True)
    
    # Adjust layout to fit legend
    plt.tight_layout()
    
    # Save the figure to the dictionary
    figures[pollutant] = fig

# Show all figures
for fig in figures.values():
    plt.show()
