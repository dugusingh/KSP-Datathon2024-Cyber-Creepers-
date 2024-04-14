import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

def predictfun():
    # Read data from CSVc
    data_prophet_df_final = pd.read_csv('/Users/navtejkumarsingh/Downloads/crime-website-main/actualdata2.csv')

    # Convert 'ds' column to datetime
    data_prophet_df_final['ds'] = pd.to_datetime(data_prophet_df_final['ds'])

    # Fit ARIMA model
    model = sm.tsa.ARIMA(data_prophet_df_final['y'], order=(1,1,1))
    results = model.fit()

    # Forecast
    forecast = results.forecast(steps=365)

    # Create future dates
    last_date = data_prophet_df_final['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date, periods=365, freq='D')

    # Combine future dates with forecast
    forecast_df = pd.DataFrame({'ds': future_dates, 'yhat': forecast})

    # Create directory if it doesn't exist
    directory = 'static/assets/data'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save forecast to JSON
    filename = os.path.join(directory, 'predictoutjsondata.json')
    forecast_df.to_json(filename, orient='records')

    # Plot forecast
    plt.figure(figsize=(10, 6))
    plt.plot(data_prophet_df_final['ds'], data_prophet_df_final['y'], label='Actual')
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
    plt.xlabel('Date')
    plt.ylabel('Crime Rate')
    plt.title('Forecasted Crime Rate')
    plt.legend()
    plt.savefig('/Users/navtejkumarsingh/Downloads/crime-website-main/static/assets/images/predictionresult.png')

    plt.show()

predictfun()