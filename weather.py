import requests
import logging
from datetime import datetime, timedelta
import pytz
import json
import os
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('weather_module')

# Configuration and Cache
CONFIG_DIR = "config"
CACHE_FILE = os.path.join(CONFIG_DIR, "weather_cache.json")
os.makedirs(CONFIG_DIR, exist_ok=True)

# OpenWeatherMap Configuration
OPENWEATHER_API_KEY = "db49109862c39b8cb1681c81aa24f315"
DEFAULT_CITY = "Faisalabad"
DEFAULT_COUNTRY_CODE = "PK"

# Air Quality Health Categories
AQI_HEALTH_CATEGORIES = [
    {
        "min": 0, "max": 12, 
        "category": "Good", 
        "emoji": "✅", 
        "color": "#00E400",
        "advice": "Air quality is excellent. Enjoy outdoor activities without concern.",
        "risk_level": 1
    },
    {
        "min": 12.1, "max": 35.4, 
        "category": "Moderate", 
        "emoji": "😐", 
        "color": "#FFFF00",
        "advice": "Air quality is acceptable. Unusually sensitive people should consider limiting prolonged outdoor exertion.",
        "risk_level": 2
    },
    {
        "min": 35.5, "max": 55.4, 
        "category": "Unhealthy for Sensitive Groups", 
        "emoji": "⚠️", 
        "color": "#FF7E00",
        "advice": "Children, elderly, and people with respiratory diseases should limit outdoor activity. General public less likely to be affected.",
        "risk_level": 3
    },
    {
        "min": 55.5, "max": 150.4, 
        "category": "Unhealthy", 
        "emoji": "❌", 
        "color": "#FF0000",
        "advice": "Health warning: Everyone may begin to experience health effects. Reduce prolonged or heavy outdoor exertion.",
        "risk_level": 4
    },
    {
        "min": 150.5, "max": 250.4, 
        "category": "Very Unhealthy", 
        "emoji": "☠️", 
        "color": "#8F3F97",
        "advice": "Health alert: Serious health effects. Avoid all outdoor activities and stay indoors with air purification.",
        "risk_level": 5
    },
    {
        "min": 250.5, 
        "max": float('inf'), 
        "category": "Hazardous", 
        "emoji": "🚨", 
        "color": "#7E0023",
        "advice": "Emergency health warning: Extreme health risks. Remain indoors with HEPA air filtration.",
        "risk_level": 6
    }
]

class WeatherService:
    def __init__(self, city=DEFAULT_CITY, country_code=DEFAULT_COUNTRY_CODE):
        """
        Initialize WeatherService with city and country details
        
        Args:
            city (str): City name
            country_code (str): Two-letter country code
        """
        self.city = city
        self.country_code = country_code
        self.api_key = OPENWEATHER_API_KEY
        
    def _load_cached_data(self):
        """Load cached weather data if available and not expired"""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, 'r') as f:
                    cached_data = json.load(f)
                    
                # Check if cache is less than 1 hour old
                cached_time = datetime.fromisoformat(cached_data.get('timestamp', '1990-01-01'))
                if datetime.now() - cached_time < timedelta(hours=1):
                    return cached_data.get('weather_data')
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
        return None
    
    def _save_cache(self, weather_data):
        """Save weather data to cache"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'weather_data': weather_data
            }
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    def get_weather_data(self):
        """
        Fetch current weather data with comprehensive details
        
        Returns:
            dict: Comprehensive weather information
        """
        # Try cache first
        cached_data = self._load_cached_data()
        if cached_data:
            return cached_data
        
        try:
            # Fetch current weather
            url = f"http://api.openweathermap.org/data/2.5/weather?q={self.city},{self.country_code}&appid={self.api_key}&units=metric"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Fetch forecast for more context
                forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={self.city},{self.country_code}&appid={self.api_key}&units=metric"
                forecast_response = requests.get(forecast_url, timeout=10)
                forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
                
                # Comprehensive weather details
                weather_data = {
                    "current": {
                        "temp": round(data["main"]["temp"], 1),
                        "feels_like": round(data["main"]["feels_like"], 1),
                        "temp_min": round(data["main"]["temp_min"], 1),
                        "temp_max": round(data["main"]["temp_max"], 1),
                        "humidity": data["main"]["humidity"],
                        "pressure": data["main"]["pressure"],
                        "condition": data["weather"][0]["description"].title(),
                        "icon": f"http://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png"
                    },
                    "wind": {
                        "speed": round(data["wind"]["speed"], 1),
                        "direction": self._get_wind_direction(data["wind"]["deg"]) if "deg" in data["wind"] else "N/A"
                    },
                    "location": {
                        "city": data["name"],
                        "country": data["sys"]["country"],
                        "timezone": pytz.FixedOffset(data["timezone"] // 60)
                    },
                    "sun": {
                        "sunrise": datetime.fromtimestamp(data["sys"]["sunrise"], tz=pytz.UTC).astimezone(pytz.FixedOffset(data["timezone"] // 60)).strftime("%I:%M %p"),
                        "sunset": datetime.fromtimestamp(data["sys"]["sunset"], tz=pytz.UTC).astimezone(pytz.FixedOffset(data["timezone"] // 60)).strftime("%I:%M %p")
                    }
                }
                
                # Add hourly forecast if available
                if forecast_data and 'list' in forecast_data:
                    weather_data['forecast'] = self._process_forecast(forecast_data['list'])
                
                # Cache the data
                self._save_cache(weather_data)
                
                return weather_data
            else:
                logger.error(f"Weather API error: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching weather: {e}")
            return None
    
    def _get_wind_direction(self, degrees):
        """Convert wind degrees to cardinal direction"""
        directions = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", 
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        return directions[int((degrees + 11.25) % 360 / 22.5)]
    
    def _process_forecast(self, forecast_list, hours=24):
        """
        Process hourly forecast data
        
        Args:
            forecast_list (list): List of forecast entries
            hours (int): Number of hours to forecast
        
        Returns:
            list: Processed forecast data
        """
        processed_forecast = []
        now = datetime.now(pytz.UTC)
        
        for entry in forecast_list[:int(hours/3)]:  # OpenWeather provides 3-hour intervals
            forecast_time = datetime.fromtimestamp(entry['dt'], tz=pytz.UTC)
            
            processed_forecast.append({
                "time": forecast_time.strftime("%Y-%m-%d %H:%M"),
                "temp": round(entry['main']['temp'], 1),
                "condition": entry['weather'][0]['description'].title(),
                "icon": f"http://openweathermap.org/img/wn/{entry['weather'][0]['icon']}@2x.png"
            })
        
        return processed_forecast
    
    def get_health_advice(self, dust_value):
        """
        Provide comprehensive health advice based on air quality
        
        Args:
            dust_value (float or str): PM2.5 concentration
        
        Returns:
            dict: Comprehensive health advice
        """
        try:
            dust = float(dust_value)
            
            # Find matching AQI category
            for category in AQI_HEALTH_CATEGORIES:
                if category['min'] <= dust <= category.get('max', float('inf')):
                    return {
                        "category": category['category'],
                        "emoji": category['emoji'],
                        "color": category['color'],
                        "advice": category['advice'],
                        "risk_level": category['risk_level'],
                        "value": dust
                    }
            
            # Fallback for very high values
            return AQI_HEALTH_CATEGORIES[-1]
        
        except (ValueError, TypeError):
            return {
                "category": "Unknown",
                "emoji": "❓",
                "color": "#CCCCCC",
                "advice": "Unable to assess health risk due to invalid data.",
                "risk_level": 0,
                "value": None
            }
    
    def plot_hourly_temperature_forecast(self):
        """
        Create an interactive temperature forecast chart
        
        Returns:
            Plotly Figure object
        """
        weather_data = self.get_weather_data()
        if not weather_data or 'forecast' not in weather_data:
            return go.Figure().update_layout(title="No forecast data available")
        
        forecast = weather_data['forecast']
        
        # Create DataFrame for plotting
        df = pd.DataFrame(forecast)
        df['time'] = pd.to_datetime(df['time'])
        
        # Create line plot
        fig = px.line(
            df, 
            x='time', 
            y='temp', 
            title='Hourly Temperature Forecast',
            labels={'time': 'Time', 'temp': 'Temperature (°C)'}
        )
        
        # Customize layout
        fig.update_layout(
            template='plotly_white',
            xaxis_title='Time',
            yaxis_title='Temperature (°C)',
            hovermode='x unified'
        )
        
        return fig
    
    def generate_weather_summary(self):
        """
        Generate a comprehensive weather summary
        
        Returns:
            dict: Detailed weather summary
        """
        weather_data = self.get_weather_data()
        if not weather_data:
            return {"status": "error", "message": "Unable to fetch weather data"}
        
        current = weather_data['current']
        wind = weather_data['wind']
        sun = weather_data.get('sun', {})
        
        # Temperature interpretation
        def interpret_temperature(temp):
            if temp < 10:
                return "Cold ❄️"
            elif 10 <= temp < 20:
                return "Cool 🧥"
            elif 20 <= temp < 30:
                return "Mild 🌞"
            else:
                return "Hot 🔥"
        
        # Humidity interpretation
        def interpret_humidity(humidity):
            if humidity < 30:
                return "Very Dry 🏜️"
            elif 30 <= humidity < 50:
                return "Dry 💨"
            elif 50 <= humidity < 70:
                return "Comfortable 😊"
            else:
                return "Humid 💦"
        
        summary = {
            "temp": {
                "value": current['temp'],
                "feels_like": current['feels_like'],
                "interpretation": interpret_temperature(current['temp'])
            },
            "humidity": {
                "value": current['humidity'],
                "interpretation": interpret_humidity(current['humidity'])
            },
            "wind": {
                "speed": wind['speed'],
                "direction": wind['direction']
            },
            "conditions": {
                "description": current['condition'],
                "icon": current['icon']
            },
            "sun": {
                "sunrise": sun.get('sunrise', 'N/A'),
                "sunset": sun.get('sunset', 'N/A')
            }
        }
        
        return summary

# Convenience function for quick access
def get_city_weather(city=DEFAULT_CITY, country_code=DEFAULT_COUNTRY_CODE):
    """
    Quick method to get weather for a specific city
    
    Args:
        city (str): City name
        country_code (str): Two-letter country code
    
    Returns:
        dict: Weather data for the specified city
    """
    service = WeatherService(city, country_code)
    return service.get_weather_data()