import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('air_quality_utils')

# ThingSpeak channel config
CHANNEL_ID = "Paste Here Your Channel ID"
READ_API_KEY = "Paste Here Your API Key"

# AQI breakpoints for PM2.5 (24-hour average)
AQI_BREAKPOINTS = [
    {"min": 0, "max": 12, "category": "Good", "color": "#00E400", "emoji": "😊", "description": "Air quality is satisfactory, and air pollution poses little or no risk."},
    {"min": 12.1, "max": 35.4, "category": "Moderate", "color": "#FFFF00", "emoji": "😐", "description": "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."},
    {"min": 35.5, "max": 55.4, "category": "Unhealthy for Sensitive Groups", "color": "#FF7E00", "emoji": "😷", "description": "Members of sensitive groups may experience health effects. The general public is less likely to be affected."},
    {"min": 55.5, "max": 150.4, "category": "Unhealthy", "color": "#FF0000", "emoji": "🤒", "description": "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."},
    {"min": 150.5, "max": 250.4, "category": "Very Unhealthy", "color": "#8F3F97", "emoji": "😫", "description": "Health alert: The risk of health effects is increased for everyone."},
    {"min": 250.5, "max": 500, "category": "Hazardous", "color": "#7E0023", "emoji": "☠️", "description": "Health warning of emergency conditions: everyone is more likely to be affected."}
]

def fetch_latest_data():
    """Fetch the most recent sensor data from ThingSpeak."""
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results=1"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and 'feeds' in data and len(data['feeds']) > 0:
                feed = data['feeds'][0]
                return {
                    'timestamp': feed.get('created_at'),
                    'temperature': safe_float(feed.get('field1')),
                    'humidity': safe_float(feed.get('field2')),
                    'mq135': safe_float(feed.get('field3')),
                    'mq7': safe_float(feed.get('field4')),
                    'dust': safe_float(feed.get('field5'))
                }
        logger.warning(f"Failed to fetch data: HTTP {response.status_code}")
        return None
    except requests.exceptions.Timeout:
        logger.error("Request timed out when fetching data from ThingSpeak")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

def fetch_historical_data(days=1, results=100):
    """Fetch historical sensor data from ThingSpeak for the specified number of days."""
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={results}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if 'feeds' in data:
                df = pd.DataFrame(data['feeds'])
                # Convert column types
                for col in ['field1', 'field2', 'field3', 'field4', 'field5']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Rename columns for clarity
                df = df.rename(columns={
                    'field1': 'temperature',
                    'field2': 'humidity',
                    'field3': 'mq135',
                    'field4': 'mq7',
                    'field5': 'dust'
                })
                
                # Convert timestamp to datetime
                df['created_at'] = pd.to_datetime(df['created_at'])
                
                return df
        logger.warning(f"Failed to fetch historical data: HTTP {response.status_code}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return pd.DataFrame()

def get_aqi_category(dust_value):
    """Convert dust value into AQI category with detailed information."""
    dust = safe_float(dust_value)
    
    if dust is None:
        return {
            "category": "Unknown",
            "label": "Unknown",
            "color": "#CCCCCC",
            "emoji": "❓",
            "description": "Unable to determine air quality due to missing data."
        }
    
    for bp in AQI_BREAKPOINTS:
        if bp["min"] <= dust <= bp["max"]:
            return {
                "category": bp["category"],
                "label": f"{bp['category']} {bp['emoji']} ({bp['min']}–{bp['max']} µg/m³)",
                "color": bp["color"],
                "emoji": bp["emoji"],
                "description": bp["description"],
                "value": dust
            }
    
    # If dust is above the highest breakpoint
    if dust > AQI_BREAKPOINTS[-1]["max"]:
        bp = AQI_BREAKPOINTS[-1]
        return {
            "category": bp["category"],
            "label": f"{bp['category']} {bp['emoji']} ({bp['min']}+ µg/m³)",
            "color": bp["color"],
            "emoji": bp["emoji"],
            "description": bp["description"],
            "value": dust
        }
    
    # If dust is below the lowest breakpoint (shouldn't happen with positive numbers)
    bp = AQI_BREAKPOINTS[0]
    return {
        "category": bp["category"],
        "label": f"{bp['category']} {bp['emoji']} ({bp['min']}–{bp['max']} µg/m³)",
        "color": bp["color"],
        "emoji": bp["emoji"],
        "description": bp["description"],
        "value": dust
    }

def safe_float(value):
    """Safely convert a value to float, returning None if conversion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def get_formatted_timestamp(timestamp_str):
    """Format a timestamp string into a human-readable format."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return "Unknown time"

def calculate_data_freshness(timestamp_str):
    """Calculate how fresh the data is and return appropriate message."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        now = datetime.utcnow()
        delta = now - dt
        
        if delta < timedelta(minutes=5):
            return "Real-time data (< 5 min old)", "green"
        elif delta < timedelta(hours=1):
            return f"Recent data ({int(delta.total_seconds() / 60)} min old)", "orange"
        else:
            return f"Stale data ({int(delta.total_seconds() / 3600)} hours old)", "red"
    except:
        return "Unknown timestamp", "gray"

def interpret_gas_sensor(mq135_value, mq7_value):
    """Interpret gas sensor readings and provide meaning."""
    mq135 = safe_float(mq135_value)
    mq7 = safe_float(mq7_value)
    
    result = {
        "mq135": {"status": "Unknown", "description": "Unable to interpret data"},
        "mq7": {"status": "Unknown", "description": "Unable to interpret data"}
    }
    
    # MQ135 - Air Quality Sensor (Ammonia, NOx, Alcohol, Benzene, Smoke, CO2)
    if mq135 is not None:
        if mq135 < 100:
            result["mq135"] = {
                "status": "Good",
                "description": "Low levels of harmful gases detected"
            }
        elif mq135 < 300:
            result["mq135"] = {
                "status": "Moderate",
                "description": "Moderate levels of harmful gases detected"
            }
        else:
            result["mq135"] = {
                "status": "Poor",
                "description": "High levels of harmful gases detected"
            }
    
    # MQ7 - Carbon Monoxide Sensor
    if mq7 is not None:
        if mq7 < 50:
            result["mq7"] = {
                "status": "Good",
                "description": "Low carbon monoxide levels"
            }
        elif mq7 < 150:
            result["mq7"] = {
                "status": "Moderate",
                "description": "Moderate carbon monoxide levels"
            }
        else:
            result["mq7"] = {
                "status": "Poor",
                "description": "High carbon monoxide levels - take precautions"
            }
    
    return result

def mock_data_if_needed(real_data):
    """Generate mock data if real data is unavailable for testing."""
    if real_data is not None:
        return real_data
        
    logger.warning("Using mock data as real data is unavailable")
    return {
        'timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        'temperature': 25.4,
        'humidity': 65.2,
        'mq135': 120.5,
        'mq7': 43.8,
        'dust': 18.7
    }
