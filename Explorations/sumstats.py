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

# List of pollutants to visualize
pollutants = ['Pm2.5', 'Pm10', 'So2', 'Co2', 'Co', 'So4']

# Initialize a list to store summary statistics
summary_stats = []

# Compute summary statistics for each DataFrame
for df_name, df in dataframes.items():
    for pollutant in pollutants:
        if pollutant in df.columns:
            stats = {
                'State': df_name,
                'Pollutant': pollutant,
                'Mean': df[pollutant].mean(),
                'Median': df[pollutant].median(),
                'Std_Dev': df[pollutant].std(),
                'Min': df[pollutant].min(),
                'Max': df[pollutant].max()
            }
            summary_stats.append(stats)

# Convert summary statistics to a DataFrame
summary_df = pd.DataFrame(summary_stats)

# Print the summary statistics DataFrame
print(summary_df)


# Unique pollutants
pollutants = summary_df['Pollutant'].unique()

# Set up the plot
plt.figure(figsize=(16, 12))

# Plot each pollutant's statistics
for i, pollutant in enumerate(pollutants):
    plt.subplot(len(pollutants), 1, i + 1)  # Create a subplot for each pollutant
    subset = summary_df[summary_df['Pollutant'] == pollutant]
    
    # Plot Mean
    plt.bar(subset['State'], subset['Mean'], width=0.4, label='Mean', alpha=0.6)
    # Plot Median
    plt.bar(subset['State'], subset['Median'], width=0.4, label='Median', alpha=0.6, hatch='//')
    # Plot Std Dev (with error bars)
    plt.errorbar(subset['State'], subset['Mean'], yerr=subset['Std_Dev'], fmt='o', label='Std Dev', color='black')

    # Add Min and Max as scatter points
    plt.scatter(subset['State'], subset['Min'], color='red', label='Min', zorder=5)
    plt.scatter(subset['State'], subset['Max'], color='blue', label='Max', zorder=5)

    # Add labels and title
    plt.ylabel('Values')
    plt.title(f'Summary Statistics for {pollutant}')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    plt.grid(True)

# Adjust layout to fit legend and labels
plt.tight_layout()

# Show the plot
plt.show()
