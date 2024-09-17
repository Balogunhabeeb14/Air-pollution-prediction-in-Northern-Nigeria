# Air Pollution Prediction Analysis

## Overview

This repository contains an analysis of air pollution prediction in the 19 northern states of Nigeria. The analysis involves predicting various pollutants using time series forecasting methods. The dataset includes information on pollutants, rain, minimum temperature, maximum temperature, and wind speed for the 19 states. Two forecasting models—LSTM (Long Short-Term Memory) and Prophet—were applied to predict pollutant levels. The results, performance metrics, and visualizations are provided in this repository.

## Data Description

The dataset consists of:
- **Pollutants**: PM2.5, PM10, SO2, CO2, CO, SO4
- **Weather Variables**: Rain, Min Temperature, Max Temperature, Wind Speed
- **Geographical Coverage**: 19 northern states of Nigeria

### Data Files
- `data/`: Contains the raw `.xlsx` files with historical data for pollutants and weather variables.

## Methodology

### Data Preparation

1. **Data Loading**: The data from `.xlsx` files is read into DataFrames and resampled to a monthly frequency.
2. **Feature Engineering**: Relevant features are selected, and missing data is handled.
3. **Scaling**: Data is normalized using `MinMaxScaler` for model training.

### Forecasting Models

1. **LSTM Model**: 
   - Uses a sequence of past observations to predict future pollutant levels.
   - The LSTM model architecture includes multiple LSTM layers with dropout for regularization.
   - Model training is performed with early stopping to prevent overfitting.

2. **Prophet Model**:
   - Utilizes time series decomposition for forecasting.
   - Handles seasonality and trend components effectively.


## Results

### Performance Metrics

Performance metrics for each model and pollutant are computed, including:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R-squared**

### Visualizations

Visualizations include:
- **Pollutant Forecasting Plots**: Comparison of actual vs. predicted pollutant levels.
- **Geographical Map**: Map of the 19 northern states of Nigeria with visualizations of pollutant levels.

## Folder Structure

- `exploration/`: Contains visualizations, including the map of the 19 northern states and other exploratory charts.
- `lstm/`: Contains scripts for training and evaluating the LSTM model, performance metrics CSV, and related charts.
- `prophet/`: Contains scripts for training and evaluating the Prophet model, performance metrics CSV, and related charts.
- `data/`: Raw data files in `.xlsx` format.

## How to Run

1. **Set Up**: Ensure you have the required Python packages installed (`pandas`, `numpy`, `scikit-learn`, `tensorflow`, `fbprophet`, `matplotlib`).

2. **Run LSTM Model**:
   ```bash
   python lstm/lstm_model.py
