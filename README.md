# Time Series Forecasting for Price Prediction

## Overview

This project focuses on forecasting the price of a product using time series analysis techniques. It employs three different models: ARMA, LSTM, and GRU, to predict future prices based on historical data. The project is implemented in Python using popular libraries such as Pandas, Statsmodels, Scikit-learn, and TensorFlow.

## Project Structure

The project is structured as follows:

1. **Data Loading and Exploration:** The code starts by loading the dataset from a CSV file ('raw_sales.csv') and performs initial data exploration to understand its structure, statistics, and potential patterns.

2. **Data Preprocessing:** The data is preprocessed by scaling the price feature using MinMaxScaler and creating a lagged feature to capture temporal dependencies.

3. **Model Building:**
    - **ARMA:** An ARIMA model (with p=2, q=2) is created and trained on the preprocessed data.
    - **LSTM:** An LSTM model is built with a hyperparameter tuning loop.
    - **GRU:** A GRU model is trained on the data.

4. **Model Evaluation:** The models are evaluated using the Root Mean Squared Error (RMSE) metric. The model with the lowest RMSE is considered the best performing model.

5. **Visualization:** The results are visualized using various plots, including:
    - Time Series Plot of Actual vs. Predicted Values
    - Residual Plots
    - Distribution of Residuals (Histograms)
    - ACF and PACF Plots (for ARMA)
![image](https://github.com/user-attachments/assets/c3ca104f-baec-4947-8f0d-f190915078aa)

## Logic and Algorithms

- **ARMA:** The ARMA model leverages autoregressive (AR) and moving average (MA) components to capture patterns in the time series data.
- **LSTM:** The LSTM model, a type of recurrent neural network (RNN), is capable of learning long-term dependencies in sequential data, making it suitable for time series forecasting.
- **GRU:** Similar to LSTM, the GRU model is another type of RNN that can effectively capture temporal dependencies, but with a simpler architecture.

## Technology Used

- **Python:** The primary programming language used for the project.
- **Pandas:** For data manipulation and analysis.
- **Statsmodels:** For implementing the ARMA model.
- **Scikit-learn:** For data scaling and model evaluation.
- **TensorFlow:** For building and training the LSTM and GRU models.
- **Matplotlib:** For creating visualizations.

## Challenges Faced

- **Data Preprocessing:** Handling missing values, outliers, and ensuring data stationarity were crucial steps.
- **Model Selection and Tuning:** Choosing the appropriate model and finding optimal hyperparameters for LSTM and GRU required experimentation.
- **Evaluation Metrics:** Selecting relevant evaluation metrics for time series forecasting and interpreting the results was important.
- **Visualizations:** Creating informative visualizations to present the results effectively was a focus.
- **Overfitting** - Overfitting was a problem encountered during modeling, this was handled using techniques like early stopping and regularization.
