#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Weather data fetch module: Retrieve real-time weather information from OpenWeatherMap API using latitude and longitude
"""

import requests

# Replace with real-time API key
OPENWEATHER_API_KEY = "YOUR API KEY"

def get_weather(lat, lon, lang="en"):
    """
    Get real-time weather data (temperature, humidity, wind speed, weather description)
    :param lat: Latitude
    :param lon: Longitude
    :param units: "metric" = Celsius, "imperial" = Fahrenheit
    :return: dict Include weather information
    """
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",  # Use metric units for retrieval, convert later if needed
        "lang": lang
    }

    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()

        # Unit conversion
        temp_c = data["main"]["temp"]
        temp_f = temp_c * 9/5 + 32
        wind_mps = data["wind"]["speed"]
        wind_mph = wind_mps * 2.23694

        return {
            "temperature": {
                "celsius": round(temp_c, 2),
                "fahrenheit": round(temp_f, 2)
            },
            "humidity": data["main"]["humidity"],
            "wind_speed": {
                "mps": round(wind_mps, 2),
                "mph": round(wind_mph, 2)
            },
            "weather_description": data["weather"][0]["description"],
            "source": "OpenWeatherMap"
        }

    except Exception as e:
        print(f"❌ Weather query failed: {e}")
        return {
            "temperature": {"celsius": None, "fahrenheit": None},
            "humidity": None,
            "wind_speed": {"mps": None, "mph": None},
            "weather_description": "Unavailable",
            "source": "OpenWeatherMap"
        }
