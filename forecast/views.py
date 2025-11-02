from django.shortcuts import render
from django.http import HttpResponse

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import os

API_KEY = '7ca13cdefafda63f41dce27482aae6af'
BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# ------------------- API Fetch -------------------
def get_forecast_weather(city):
    """Fetch forecast data for better Min/Max temperature accuracy"""
    url = f"{BASE_URL}forecast?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()

    # Get temperature data from all forecast entries
    temps = [entry['main']['temp'] for entry in data['list']]
    min_temp = round(min(temps))
    max_temp = round(max(temps))

    # Take the first forecast entry as "current-like" weather
    current = data['list'][0]

    return {
        'city': data['city']['name'],
        'current_temp': round(current['main']['temp']),
        'feels_like': round(current['main']['feels_like']),
        'temp_min': min_temp,
        'temp_max': max_temp,
        'humidity': round(current['main']['humidity']),
        'description': current['weather'][0]['description'],
        'country': data['city']['country'],
        'wind_gust_speed': round(current['wind']['speed']),
        'wind_gust_dir': current['wind'].get('deg', 0),
        'pressure': current['main']['pressure'],
        'clouds': current['clouds']['all'],
        'visibility': current.get('visibility', 0)
    }

# ------------------- Historical Data -------------------
def read_historical_data(filename):
    df = pd.read_csv(filename)
    df = df.dropna().drop_duplicates()
    return df

# ------------------- Prepare Data -------------------
def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    X = data.drop(['RainTomorrow'], axis=1)
    y = data['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Mean square error:", mean_squared_error(y_test, y_pred))
    return model

# ------------------- Regression Models -------------------
def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data) - 1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i + 1])
    return np.array(X).reshape(-1, 1), np.array(y)

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_value, steps=5):
    predictions = [current_value]
    for _ in range(steps):
        next_value = model.predict(np.array(predictions[-1]).reshape(-1, 1))
        predictions.append(next_value[0])
    return predictions[1:]

# ------------------- View -------------------
def weather_view(request):
    context = {
        'location': '',
        'current_temp': '--',
        'MinTemp': '--',
        'MaxTemp': '--',
        'feels_like': '--',
        'humidity': '--',
        'clouds': '--',
        'description': 'Weather Forecast',
        'country': '--',
        'time': '--',
        'date': '--',
        'wind': '--',
        'pressure': '--',
        'visibility': '--',
        'forecast': []
    }

    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_forecast_weather(city)  # ✅ Changed here

        csv_path = os.path.join('C:/MajorProject/WeatherPrediction/weather.csv')
        historical_data = read_historical_data(csv_path)
        X, y, le = prepare_data(historical_data)
        rain_model = train_rain_model(X, y)

        # Wind direction
        wind_deg = current_weather['wind_gust_dir'] % 360
        compass_points = [
            ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
            ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
            ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
            ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
            ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
            ("NNW", 326.25, 348.75), ("N", 348.75, 360)
        ]
        compass_direction = next(
            (point for point, start, end in compass_points if start <= wind_deg < end),
            "N"
        )
        compass_encoded = le.transform([compass_direction])[0] if compass_direction in le.classes_ else -1

        # Rain prediction
        current_data = pd.DataFrame([{
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'WindGustDir': compass_encoded,
            'WindGustSpeed': current_weather['wind_gust_speed'],
            'Humidity': current_weather['humidity'],
            'Pressure': current_weather['pressure'],
            'Temp': current_weather['feels_like']
        }])
        rain_prediction = rain_model.predict(current_data)[0]

        # Temperature & Humidity prediction
        X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
        temp_model = train_regression_model(X_temp, y_temp)
        temp_pred = predict_future(temp_model, current_weather['temp_min'])

        X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
        hum_model = train_regression_model(X_hum, y_hum)
        hum_pred = predict_future(hum_model, current_weather['humidity'])

        # Future times
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # Forecast list
        forecast = []
        for i in range(5):
            forecast.append({
                'time': future_times[i],
                'temp': round(temp_pred[i], 1),
                'hum': round(hum_pred[i], 1)
            })

        context = {
            'location': city,
            'current_temp': current_weather['current_temp'],
            'MinTemp': current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description': current_weather['description'],
            'country': current_weather['country'],
            'time': now.strftime("%H:%M"),
            'date': now.strftime("%B %d, %Y"),
            'wind': current_weather['wind_gust_speed'],
            'pressure': current_weather['pressure'],
            'visibility': current_weather['visibility'],
            'forecast': forecast
        }

    return render(request, 'weather.html', context)
