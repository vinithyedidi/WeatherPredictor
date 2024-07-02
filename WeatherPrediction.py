import numpy as np
import pandas as pd
import openmeteo_requests
from datetime import datetime, timedelta
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import VAR
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

'''
Data downloaded from the Iowa State University IEM ASOS network:
https://mesonet.agron.iastate.edu/request/download.phtml

We take weather data from Boston Logan International Airport every hour
from 1943 until 2024 to make a regression model and use it to predict
the weather for the next hour and up to 24 hours in the future. Then we 
compare the model's predictions to other forecasts and real observations. 

FEATURES:
    temp: in F
    humidity: relative humidity in %
    speed: wind speed in MPH
    pressure: sea-level air pressure in millibars
    delta_t, delta_h: hourly change in temp, humidity
    hour: used to correlate time of day to weather
'''


def process_data(filepath):

    df = pd.read_csv(filepath)
    df.drop(['station', 'direction'], axis=1, inplace=True)
    df.isnull().sum()

    df['delta_t'] = df['temp'] - df['temp'].shift(periods=-1)
    df['delta_h'] = df['humidity'] - df['humidity'].shift(periods=-1)
    df['time'] = (pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M'))
    df['hour'] = df['time'].dt.hour + 1  # since all times are at the :54, we round to next hour
    df.dropna(inplace=True)

    return df


def make_model(df):

    # We predict 18 hours ahead, with a lag of 25
    forecast_window = 18
    p = 25    # (determined to be optimal with AIC = 8.068 )
    train, test = df[:-p], df[-forecast_window:]

    # Our model is a VAR model with the endogenous variables being all the below variable except hour
    model = VAR(endog=train.loc[:, ['delta_t', 'pressure', 'speed', 'humidity', 'delta_h']],
                 exog=train.loc[:, ['hour']])
    fitted_model = model.fit(p)
    # print(result.summary())

    # Test accuracy of model's forecast of the last 18 hours of the data
    delta_t_fc = np.array(fitted_model.forecast(y=fitted_model.endog, exog_future=test.loc[:, ['hour']],
                                                steps=forecast_window))[:, 0]
    test_temps = np.array(test.values[:forecast_window, 1])
    pred_temps = np.zeros(forecast_window)
    pred_temps[0] = test_temps[0]
    for i in range(1, len(test_temps)):
        pred_temps[i] = pred_temps[i - 1] + delta_t_fc[i]
    error_metrics = {
        'RSME': root_mean_squared_error(test_temps, pred_temps),
        'MAE': mean_absolute_error(test_temps, pred_temps),
    }
    return fitted_model, error_metrics


def get_weather(past: bool):

    # We use Open-Meteo API to obtain our weather data for free.
    openmeteo = openmeteo_requests.Client()

    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': 42.3654,
        'longitude': -71.0108,
        'hourly': ['temperature_2m', 'relative_humidity_2m', 'pressure_msl', 'wind_speed_10m'],
        'start_hour': (datetime.now() - timedelta(hours=24)).replace(minute=0).strftime('%Y-%m-%dT%H:%M'),
        'end_hour': (datetime.now()).replace(minute=0).strftime('%Y-%m-%dT%H:%M'),
        'temperature_unit': 'fahrenheit',
        'wind_speed_unit': 'mph',
        'timezone': 'America/New_York'
    }
    if past:
        params['start_hour'] = (datetime.now() - timedelta(hours=27)).replace(minute=0).strftime('%Y-%m-%dT%H:%M')
        params['end_hour'] = (datetime.now() - timedelta(hours=3)).replace(minute=0).strftime('%Y-%m-%dT%H:%M')
    else:
        params['start_hour'] = (datetime.now() - timedelta(hours=2)).replace(minute=0).strftime('%Y-%m-%dT%H:%M')
        params['end_hour'] = (datetime.now() + timedelta(hours=21)).replace(minute=0).strftime('%Y-%m-%dT%H:%M')

    responses = openmeteo.weather_api(url, params=params)
    hourly = responses[0]
    temp = hourly.Hourly().Variables(0).ValuesAsNumpy()
    humidity = hourly.Hourly().Variables(1).ValuesAsNumpy()
    pressure = hourly.Hourly().Variables(2).ValuesAsNumpy()
    speed = hourly.Hourly().Variables(3).ValuesAsNumpy()
    delta_t = delta_h = np.zeros(len(humidity))
    for i in range(1, len(humidity)):
        delta_h[i] = humidity[i] - humidity[i-1]
        delta_t[i] = temp[i] - temp[i-1]
    time = pd.date_range(
        start=pd.to_datetime(hourly.Hourly().Time(), unit="s"),
        end=pd.to_datetime(hourly.Hourly().TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Hourly().Interval()),
        inclusive="left"
    )
    hours = time.hour
    weather_data = pd.DataFrame({
        'time': time,
        'temp': temp,
        'delta_t': delta_t,
        'pressure': pressure,
        'speed': speed,
        'humidity': humidity,
        'delta_h': delta_h,
        'hour': hours
    })

    return weather_data


def predict(fitted_model, weather_data, ):

    forecast_window = 24

    hours = np.empty(forecast_window)
    hours[0] = datetime.now().hour
    for i in range(1, len(hours)):
        hours[i] = hours[i-1] + 1
        if hours[i] >= 24:
            hours[i] -= 24
    pred_temps = np.empty(forecast_window)
    pred_temps[0] = weather_data.loc[0].at['temp']
    params = weather_data[['delta_t', 'pressure', 'speed', 'humidity', 'delta_h']]

    fc = np.array(fitted_model.forecast(y=np.array(params), exog_future=hours, steps=forecast_window))
    delta_t_fc = fc[:, 0]
    for i in range(1, forecast_window):
        pred_temps[i] = pred_temps[i - 1] + delta_t_fc[i]

    prediction = pd.DataFrame({
        'Hour': hours,
        'Temp': np.round(pred_temps, 1)
    })
    return prediction


def main():

    # First we process the data, generate and test a VAR model accounting for each regressor, and then
    # generate error metrics for a 24-hour prediction.
    df = process_data('/Users/vinithyedidi/PycharmProjects/WeatherPrediction/BOS_extensive.csv')
    fitted_model, error_metrics = make_model(df)

    # We make a correlation heatmap, showing that the regressors we chose are appropriately highly correlated.
    plt.figure(figsize=(16, 6))
    heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 12}, pad=12)
    plt.show()

    # We are interested in forecasting the next hour temp ex-ante, so we use our predictions for each
    # regressor for each hour to forecast the temperature for 24 hours starting with current weather.
    _24hr_weather_data = get_weather(True)
    My_Forecast = predict(fitted_model, _24hr_weather_data)
    NOAA_Forecast = get_weather(False)

    # We now tabulate the results and plot against NOAA predictions, making graphs and metrics.
    hours = My_Forecast['Hour']
    for i in range(len(hours)):
        hours[i] = str(int(hours[i]))
        hours[i]= hours[i] + (':00')
    print(tabulate(My_Forecast, headers='keys', tablefmt='simple_table', showindex=False))
    print('\nAuto Regressive Error Metrics:')
    print('RSME: %.2f' % error_metrics['RSME'])
    print('MAE:  %.2f' % error_metrics['MAE'])
    print('\nNOAA Comparative Error Metrics:')
    print('RMSE: %.2f' % root_mean_squared_error(NOAA_Forecast['temp'], My_Forecast['Temp']))
    print('MAE:  %.2f' % mean_absolute_error(NOAA_Forecast['temp'], My_Forecast['Temp']))

    plt.figure(figsize=(16,6))
    plt.title('Comparing My Model to NOAA Model over Time')
    plt.plot(hours, My_Forecast['Temp'], '-b', label='My model')
    plt.plot(hours, NOAA_Forecast['temp'], '-r', label='NOAA model')
    plt.xlabel('Time')
    plt.ylabel('Temp (F)')
    plt.legend(loc='best')
    plt.gcf().autofmt_xdate()
    plt.show()


if __name__ == '__main__':
    main()
