import requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import time
import joblib
import os
from statsmodels.tsa.arima.model import ARIMA

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('air_quality_prediction')

# ThingSpeak channel config
CHANNEL_ID = "Paste Here Your Channel ID"
READ_API_KEY = "Paste Here Your API Key"

# Model save paths
MODEL_DIR = "models"
LINEAR_MODEL_PATH = os.path.join(MODEL_DIR, "linear_model.joblib")
RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.joblib")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_past_data(results=1000, fields=None):
    """
    Fetch historical data from ThingSpeak
    
    Args:
        results: Number of results to fetch
        fields: List of field numbers to fetch (e.g., [1, 2, 5])
    
    Returns:
        Pandas DataFrame with processed data
    """
    fields_param = "" if fields is None else f"&fields={','.join(map(str, fields))}"
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.json?api_key={READ_API_KEY}&results={results}{fields_param}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if 'feeds' in data and len(data['feeds']) > 0:
                feeds = data['feeds']
                df = pd.DataFrame(feeds)
                
                # Convert timestamp
                df['created_at'] = pd.to_datetime(df['created_at'])
                
                # Convert all numeric fields
                numeric_columns = {}
                for i in range(1, 6):
                    field_name = f'field{i}'
                    if field_name in df.columns:
                        df[field_name] = pd.to_numeric(df[field_name], errors='coerce')
                        
                        # Rename columns for clarity
                        if i == 1:
                            numeric_columns[field_name] = 'temperature'
                        elif i == 2:
                            numeric_columns[field_name] = 'humidity'
                        elif i == 3:
                            numeric_columns[field_name] = 'mq135'
                        elif i == 4:
                            numeric_columns[field_name] = 'mq7'
                        elif i == 5:
                            numeric_columns[field_name] = 'dust'
                
                # Rename columns
                df = df.rename(columns=numeric_columns)
                
                # Drop rows with missing target values
                needed_columns = ['dust'] if 'dust' in df.columns else []
                if needed_columns:
                    df = df.dropna(subset=needed_columns)
                
                # Add timestamp features
                df['timestamp'] = df['created_at'].astype('int64') // 10**9
                df['hour'] = df['created_at'].dt.hour
                df['day_of_week'] = df['created_at'].dt.dayofweek
                
                return df
            else:
                logger.warning("No data found in ThingSpeak response")
                return None
        else:
            logger.error(f"Failed to fetch data: HTTP {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching past data: {e}")
        return None

def prepare_features(df):
    """Prepare features for training prediction models"""
    if df is None or len(df) < 10:  # Need enough data points
        return None, None, None
    
    feature_cols = []
    
    # Add basic features
    if 'timestamp' in df.columns:
        feature_cols.append('timestamp')
    if 'hour' in df.columns:
        feature_cols.append('hour')
    if 'day_of_week' in df.columns:
        feature_cols.append('day_of_week')
    
    # Add sensor features if available
    sensor_features = ['temperature', 'humidity', 'mq135', 'mq7']
    for feature in sensor_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            if df[feature].notna().sum() > len(df) * 0.5:  # Use feature if at least 50% of values are valid
                feature_cols.append(feature)
                
    # Ensure we have target column
    if 'dust' not in df.columns or df['dust'].notna().sum() < 10:
        logger.error("Not enough valid dust measurements for prediction")
        return None, None, None
        
    # Filter to only rows with valid dust values
    df = df.dropna(subset=['dust'])
    
    # If we don't have enough features, use timestamp as minimum
    if not feature_cols:
        if 'timestamp' not in df.columns:
            df['timestamp'] = df['created_at'].astype('int64') // 10**9
        feature_cols = ['timestamp']
    
    # Get feature and target arrays
    X = df[feature_cols].values
    y = df['dust'].values
    
    return X, y, feature_cols

def train_prediction_models(force_retrain=False):
    """Train both Linear Regression and Random Forest models"""
    # Check if models already exist and we're not forcing retrain
    if not force_retrain and os.path.exists(LINEAR_MODEL_PATH) and os.path.exists(RF_MODEL_PATH):
        try:
            # If models exist and are less than 1 day old, use them
            if (time.time() - os.path.getmtime(LINEAR_MODEL_PATH) < 86400 and 
                time.time() - os.path.getmtime(RF_MODEL_PATH) < 86400):
                logger.info("Using existing models (less than 1 day old)")
                linear_model = joblib.load(LINEAR_MODEL_PATH)
                rf_model = joblib.load(RF_MODEL_PATH)
                
                # Get a small sample for feature names
                sample_df = fetch_past_data(results=10)
                if sample_df is not None:
                    _, _, feature_cols = prepare_features(sample_df)
                    return linear_model, rf_model, feature_cols
        except Exception as e:
            logger.warning(f"Error loading saved models: {e}. Training new models.")
    
    # Fetch data
    df = fetch_past_data(results=1000)
    if df is None or len(df) < 10:
        logger.error("Not enough data for training models")
        return None, None, None
    
    # Prepare data
    X, y, feature_cols = prepare_features(df)
    if X is None:
        return None, None, None
    
    # Handle potential timeouts by limiting data points
    max_samples = 500
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    try:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Linear Regression model
        linear_model = LinearRegression()
        linear_model.fit(X_train_scaled, y_train)
        
        # Train Random Forest model
        rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        
        # Save models
        joblib.dump(linear_model, LINEAR_MODEL_PATH)
        joblib.dump(rf_model, RF_MODEL_PATH)
        
        # Log model performance
        lr_score = r2_score(y_test, linear_model.predict(X_test_scaled))
        rf_score = r2_score(y_test, rf_model.predict(X_test_scaled))
        logger.info(f"Model R² scores - Linear: {lr_score:.3f}, RandomForest: {rf_score:.3f}")
        
        return linear_model, rf_model, feature_cols
    
    except Exception as e:
        logger.error(f"Error training models: {e}")
        return None, None, None

def get_future_timestamp_features(last_timestamp, periods=24, freq='1H'):
    """Generate features for future timestamps"""
    future_timestamps = pd.date_range(last_timestamp, periods=periods, freq=freq)
    future_df = pd.DataFrame()
    future_df['created_at'] = future_timestamps
    future_df['timestamp'] = future_df['created_at'].astype('int64') // 10**9
    future_df['hour'] = future_df['created_at'].dt.hour
    future_df['day_of_week'] = future_df['created_at'].dt.dayofweek
    return future_df

def predict_future_values(df, periods=24, freq='1H'):
    """Predict future dust values"""
    if df is None or len(df) < 10:
        logger.warning("Not enough data for prediction")
        return None, None
    
    # Train or load models
    linear_model, rf_model, feature_cols = train_prediction_models()
    if linear_model is None or feature_cols is None:
        logger.warning("Could not train/load prediction models")
        return None, None
    
    # Get last row for starting point
    last_row = df.iloc[-1]
    
    # Features needed for future prediction
    future_df = get_future_timestamp_features(last_row['created_at'], periods, freq)
    
    # For sensor values, we'll use the last known values
    # (In a real system, you'd want to predict these too, but this is a simplification)
    for col in feature_cols:
        if col not in future_df.columns and col in df.columns:
            future_df[col] = df[col].iloc[-1]  # Use last known value
    
    # Ensure we have all needed features
    for col in feature_cols:
        if col not in future_df.columns:
            if col == 'timestamp':
                future_df['timestamp'] = future_df['created_at'].astype('int64') // 10**9
            else:
                logger.warning(f"Missing feature for prediction: {col}")
                return None, None
    
    # Extract features
    future_X = future_df[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    current_X = df[feature_cols].values
    scaler.fit(current_X)
    future_X_scaled = scaler.transform(future_X)
    
    # Make predictions with both models
    future_linear_preds = linear_model.predict(future_X_scaled)
    future_rf_preds = rf_model.predict(future_X_scaled)
    
    # Ensure no negative predictions
    future_linear_preds = np.maximum(0, future_linear_preds)
    future_rf_preds = np.maximum(0, future_rf_preds)
    
    return future_df['created_at'], future_linear_preds, future_rf_preds

def plot_prediction_chart(prediction_hours=24):
    """Create interactive chart with historical data and predictions"""
    # Fetch data
    df = fetch_past_data(1000)
    if df is None or len(df) < 10:
        return go.Figure().update_layout(
            title="Insufficient data for prediction",
            annotations=[{
                "text": "Not enough data available from sensors",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
    
    # Generate predictions
    future_timestamps, linear_preds, rf_preds = predict_future_values(df, periods=prediction_hours)
    
    if future_timestamps is None:
        return go.Figure().update_layout(
            title="Unable to generate predictions",
            annotations=[{
                "text": "Could not generate predictions from available data",
                "showarrow": False,
                "font": {"size": 20}
            }]
        )
    
    # Create prediction visualization
    fig = go.Figure()
    
    # Plot historical data - show limited recent data for clarity
    history_limit = min(100, len(df))
    historical_data = df.iloc[-history_limit:]
    
    # Add Historical Data
    fig.add_trace(go.Scatter(
        x=historical_data['created_at'], 
        y=historical_data['dust'],
        mode='lines+markers',
        name="Historical Data",
        line=dict(color="#1F77B4", width=2)
    ))
    
    # Add Linear Regression Prediction
    fig.add_trace(go.Scatter(
        x=future_timestamps, 
        y=linear_preds,
        mode='lines',
        name="Linear Prediction",
        line=dict(dash='dot', color="#FF7F0E", width=2)
    ))
    
    # Add Random Forest Prediction
    fig.add_trace(go.Scatter(
        x=future_timestamps, 
        y=rf_preds,
        mode='lines',
        name="ML Prediction",
        line=dict(dash='dash', color="#2CA02C", width=2)
    ))
    
    # Calculate average prediction
    avg_preds = (linear_preds + rf_preds) / 2
    fig.add_trace(go.Scatter(
        x=future_timestamps, 
        y=avg_preds,
        mode='lines+markers',
        name="Ensemble Prediction",
        line=dict(color="#D62728", width=3)
    ))
    
    # Add confidence interval (simplified)
    upper_bound = avg_preds * 1.2
    lower_bound = avg_preds * 0.8
    
    fig.add_trace(go.Scatter(
        x=list(future_timestamps) + list(future_timestamps)[::-1],
        y=list(upper_bound) + list(lower_bound)[::-1],
        fill='toself',
        fillcolor='rgba(214, 39, 40, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Prediction Interval',
        showlegend=True
    ))
    
    # Format the chart
    fig.update_layout(
        title="Air Quality Prediction - PM2.5 (μg/m³)",
        xaxis_title="Time",
        yaxis_title="PM2.5 (μg/m³)",
        legend_title="Data Source",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
   # Calculate y1 (vertical line height)
    max_dust = historical_data['dust'].max() * 1.1
    max_upper = np.max(upper_bound) * 1.1
    y1 = max(max_dust, max_upper)

    # Now use it in the shape
    fig.add_shape(
        type="line",
        x0=historical_data['created_at'].iloc[-1],
        y0=0,
        x1=historical_data['created_at'].iloc[-1],
        y1=y1,
        line=dict(
            color="Gray",
            width=2,
            dash="dash",
        )
    )

    
    # Add annotation for prediction start
    fig.add_annotation(
        x=historical_data['created_at'].iloc[-1],
        y=historical_data['dust'].iloc[-1],
        text="Prediction starts",
        showarrow=True,
        arrowhead=1,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="gray"
    )
    
    return fig

def generate_forecast_metrics(hours_ahead=24):
    """Generate forecast metrics and summary for display"""
    # Fetch data
    df = fetch_past_data(500)
    if df is None or len(df) < 10:
        return {
            "status": "error",
            "message": "Insufficient data for forecast"
        }
    
    # Get predictions
    future_timestamps, linear_preds, rf_preds = predict_future_values(df, periods=hours_ahead)
    if future_timestamps is None:
        return {
            "status": "error",
            "message": "Unable to generate forecast"
        }
    
    # Calculate ensemble prediction
    ensemble_preds = (linear_preds + rf_preds) / 2
    
    # AQI categories thresholds (simplified)
    aqi_thresholds = [
        {"max": 12, "category": "Good", "color": "#00E400"},
        {"max": 35.4, "category": "Moderate", "color": "#FFFF00"},
        {"max": 55.4, "category": "Unhealthy for Sensitive Groups", "color": "#FF7E00"},
        {"max": 150.4, "category": "Unhealthy", "color": "#FF0000"},
        {"max": 250.4, "category": "Very Unhealthy", "color": "#8F3F97"},
        {"max": float('inf'), "category": "Hazardous", "color": "#7E0023"}
    ]
    
    # Find forecast categories
    forecast_categories = []
    for value in ensemble_preds:
        for threshold in aqi_thresholds:
            if value <= threshold["max"]:
                forecast_categories.append({
                    "category": threshold["category"],
                    "color": threshold["color"]
                })
                break
    
    # Current AQI
    current_dust = df['dust'].iloc[-1]
    current_category = None
    for threshold in aqi_thresholds:
        if current_dust <= threshold["max"]:
            current_category = threshold["category"]
            current_color = threshold["color"]
            break
    
    # Trend calculation
    dust_values = df['dust'].iloc[-24:] if len(df) >= 24 else df['dust']
    trend_direction = "stable"
    trend_emoji = "➡️"
    
    if len(dust_values) >= 3:
        # Simple trend detection
        first_half = dust_values.iloc[:len(dust_values)//2].mean()
        second_half = dust_values.iloc[len(dust_values)//2:].mean()
        
        if second_half > first_half * 1.1:  # 10% increase
            trend_direction = "worsening"
            trend_emoji = "⬆️"
        elif second_half < first_half * 0.9:  # 10% decrease
            trend_direction = "improving"
            trend_emoji = "⬇️"
    
    # Max predicted value and when
    max_pred = max(ensemble_preds)
    max_index = np.argmax(ensemble_preds)
    max_time = future_timestamps[max_index].strftime("%Y-%m-%d %H:%M")
    
    # Min predicted value and when
    min_pred = min(ensemble_preds)
    min_index = np.argmin(ensemble_preds)
    min_time = future_timestamps[min_index].strftime("%Y-%m-%d %H:%M")
    
    # Calculate most frequent category
    category_counts = {}
    most_common_category = {"category": "Unknown", "count": 0}
    
    for cat in forecast_categories:
        if cat["category"] not in category_counts:
            category_counts[cat["category"]] = 0
        category_counts[cat["category"]] += 1
        
        if category_counts[cat["category"]] > most_common_category["count"]:
            most_common_category = {
                "category": cat["category"],
                "count": category_counts[cat["category"]],
                "color": cat["color"]
            }
    
    # Prepare summary
    forecast_summary = {
        "status": "success",
        "current": {
            "value": round(current_dust, 1),
            "category": current_category,
            "color": current_color
        },
        "trend": {
            "direction": trend_direction,
            "emoji": trend_emoji
        },
        "forecast": {
            "hours": hours_ahead,
            "average": round(np.mean(ensemble_preds), 1),
            "max": {
                "value": round(max_pred, 1),
                "time": max_time
            },
            "min": {
                "value": round(min_pred, 1),
                "time": min_time
            },
            "dominant_category": most_common_category
        }
    }
    
    return forecast_summary

def plot_aqi_distribution_forecast():
    """Create a visualization of forecasted AQI category distribution"""
    # Fetch data
    df = fetch_past_data(500)
    if df is None or len(df) < 10:
        return go.Figure().update_layout(title="Insufficient data for AQI forecast")
    
    # Get predictions for next 24 hours
    future_timestamps, linear_preds, rf_preds = predict_future_values(df, periods=24)
    if future_timestamps is None:
        return go.Figure().update_layout(title="Unable to generate AQI forecast")
    
    # Calculate ensemble prediction
    ensemble_preds = (linear_preds + rf_preds) / 2
    
    # AQI categories
    aqi_categories = [
        {"max": 12, "category": "Good", "color": "#00E400"},
        {"max": 35.4, "category": "Moderate", "color": "#FFFF00"},
        {"max": 55.4, "category": "Unhealthy for Sensitive Groups", "color": "#FF7E00"},
        {"max": 150.4, "category": "Unhealthy", "color": "#FF0000"},
        {"max": 250.4, "category": "Very Unhealthy", "color": "#8F3F97"},
        {"max": float('inf'), "category": "Hazardous", "color": "#7E0023"}
    ]
    
    # Count predictions in each category
    category_counts = {cat["category"]: 0 for cat in aqi_categories}
    
    for pred in ensemble_preds:
        for cat in aqi_categories:
            if pred <= cat["max"]:
                category_counts[cat["category"]] += 1
                break
    
    # Create category distribution bar chart
    categories = []
    counts = []
    colors = []
    
    for cat in aqi_categories:
        if category_counts[cat["category"]] > 0:
            categories.append(cat["category"])
            counts.append(category_counts[cat["category"]])
            colors.append(cat["color"])
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=categories,
        values=counts,
        marker=dict(colors=colors),
        hole=.3,
        textinfo='percent',
        hoverinfo='label+value+percent'
    )])
    
    fig.update_layout(
        title="24-Hour AQI Forecast Distribution",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    
    return fig

def plot_hourly_forecast():
    """Create hourly forecast chart for next 24 hours"""
    # Fetch data
    df = fetch_past_data(500)
    if df is None or len(df) < 10:
        return go.Figure().update_layout(title="Insufficient data for hourly forecast")
    
    # Get predictions for next 24 hours
    future_timestamps, _, rf_preds = predict_future_values(df, periods=24, freq='1H')
    if future_timestamps is None:
        return go.Figure().update_layout(title="Unable to generate hourly forecast")
    
    # Format x-axis labels for hours
    x_labels = [ts.strftime('%H:%M') for ts in future_timestamps]
    
    # Get color for each prediction based on AQI category
    colors = []
    for pred in rf_preds:
        if pred <= 12:
            colors.append("#00E400")  # Good
        elif pred <= 35.4:
            colors.append("#FFFF00")  # Moderate
        elif pred <= 55.4:
            colors.append("#FF7E00")  # Unhealthy for Sensitive Groups
        elif pred <= 150.4:
            colors.append("#FF0000")  # Unhealthy
        elif pred <= 250.4:
            colors.append("#8F3F97")  # Very Unhealthy
        else:
            colors.append("#7E0023")  # Hazardous
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=x_labels,
        y=rf_preds,
        marker_color=colors,
        text=[f"{round(val, 1)} µg/m³" for val in rf_preds],
        textposition='auto',
        hovertemplate='%{x}: %{y:.1f} µg/m³<extra></extra>'
    ))
    
    # Add reference lines for AQI thresholds
    fig.add_shape(
        type="line", line=dict(dash="dash", color="green"),
        x0=0, x1=1, y0=12, y1=12,
        xref="paper", yref="y"
    )
    
    fig.add_shape(
        type="line", line=dict(dash="dash", color="orange"),
        x0=0, x1=1, y0=35.4, y1=35.4,
        xref="paper", yref="y"
    )
    
    fig.update_layout(
        title="Hourly PM2.5 Forecast",
        xaxis_title="Hour",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_white",
        xaxis=dict(tickangle=45),
        yaxis=dict(
            rangemode="tozero",
            dtick=10
        )
    )
    
    # Add annotations for AQI thresholds
    fig.add_annotation(
        x=0, y=12,
        xref="paper", yref="y",
        text="Good",
        showarrow=False,
        xanchor="left",
        bgcolor="rgba(0,228,0,0.3)",
        bordercolor="green",
        borderwidth=1
    )
    
    fig.add_annotation(
        x=0, y=35.4,
        xref="paper", yref="y",
        text="Moderate",
        showarrow=False,
        xanchor="left",
        bgcolor="rgba(255,255,0,0.3)",
        bordercolor="orange",
        borderwidth=1
    )
    
    return fig

def plot_sensor_correlations():
    """Create a correlation heatmap between different sensors"""
    # Fetch data for all sensors
    df = fetch_past_data(500)
    if df is None or len(df) < 10:
        return go.Figure().update_layout(title="Insufficient data for correlation analysis")
    
    # Select relevant columns
    sensor_cols = ['temperature', 'humidity', 'mq135', 'mq7', 'dust']
    available_cols = [col for col in sensor_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return go.Figure().update_layout(title="Not enough sensor data for correlation analysis")
    
    # Calculate correlations
    corr_df = df[available_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns,
        y=corr_df.index,
        colorscale='RdBu_r',
        zmin=-1, zmax=1,
        text=[[f"{val:.2f}" for val in row] for row in corr_df.values],
        texttemplate="%{text}",
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Sensor Correlation Analysis",
        template="plotly_white",
        height=500,
        width=600
    )
    
    return fig

def plot_comparison_chart():
    """Compare past 24h vs next 24h AQI forecast"""
    df = fetch_past_data(1000)
    if df is None or len(df) < 48:
        return go.Figure().update_layout(title="Insufficient data for comparison")

    future_timestamps, _, rf_preds = predict_future_values(df, periods=24)
    df_recent = df.iloc[-24:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_recent['created_at'], y=df_recent['dust'],
        mode='lines+markers', name='Past 24h',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=future_timestamps, y=rf_preds,
        mode='lines+markers', name='Next 24h Forecast',
        line=dict(color='green', dash='dash')
    ))

    fig.update_layout(
        title="Past vs Future PM2.5 AQI",
        xaxis_title="Time",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_mq135_prediction_chart():
    """Forecast MQ135 sensor values for the next 24 hours"""
    df = fetch_past_data(1000)
    if df is None or 'mq135' not in df.columns or df['mq135'].dropna().empty:
        return go.Figure().update_layout(title="Insufficient MQ135 data")

    df = df.dropna(subset=['mq135'])
    recent = df[['created_at', 'mq135']].iloc[-24:]
    avg_val = recent['mq135'].mean()
    timestamps = pd.date_range(start=recent['created_at'].iloc[-1], periods=24, freq='1H')
    prediction = np.clip(np.random.normal(avg_val, 5, 24), 0, None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent['created_at'], y=recent['mq135'],
                             mode='lines+markers', name='Past MQ135',
                             line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=timestamps, y=prediction,
                             mode='lines+markers', name='Predicted MQ135',
                             line=dict(dash='dot', color='red')))
    fig.update_layout(title="MQ135 Prediction (Next 24 Hours)",
                      xaxis_title="Time", yaxis_title="MQ135 Sensor Value",
                      template="plotly_white")
    return fig


def plot_mq135_dust_combined_chart():
    """Compare MQ135 and Dust Sensor Predictions"""
    df = fetch_past_data(1000)
    if df is None or len(df) < 48 or 'mq135' not in df.columns or 'dust' not in df.columns:
        return go.Figure().update_layout(title="Insufficient data for combined prediction")

    df = df.dropna(subset=['mq135', 'dust'])
    df_recent = df.iloc[-24:]

    avg_mq135 = df_recent['mq135'].mean()
    mq135_future = np.clip(np.random.normal(avg_mq135, 5, 24), 0, None)
    future_timestamps, _, dust_preds = predict_future_values(df, periods=24)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_recent['created_at'], y=df_recent['mq135'],
                             mode='lines', name='Past MQ135', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=future_timestamps, y=mq135_future,
                             mode='lines', name='Predicted MQ135', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=future_timestamps, y=dust_preds,
                             mode='lines', name='Predicted Dust', line=dict(color='green')))
    fig.update_layout(title="Combined MQ135 & Dust Prediction",
                      xaxis_title="Time", yaxis_title="Sensor Values",
                      template="plotly_white")
    return fig

def plot_mq7_prediction_chart():
    """Forecast MQ7 sensor values for the next 24 hours"""
    df = fetch_past_data(1000)
    if df is None or 'mq7' not in df.columns or df['mq7'].dropna().empty:
        return go.Figure().update_layout(title="Insufficient MQ7 data")

    df = df.dropna(subset=['mq7'])
    recent = df[['created_at', 'mq7']].iloc[-24:]
    avg_val = recent['mq7'].mean()
    timestamps = pd.date_range(start=recent['created_at'].iloc[-1], periods=24, freq='1H')
    prediction = np.clip(np.random.normal(avg_val, 3, 24), 0, None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent['created_at'], y=recent['mq7'],
                             mode='lines+markers', name='Past MQ7',
                             line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=timestamps, y=prediction,
                             mode='lines+markers', name='Predicted MQ7',
                             line=dict(dash='dot', color='darkblue')))
    fig.update_layout(title="MQ7 Prediction (Next 24 Hours)",
                      xaxis_title="Time", yaxis_title="MQ7 Sensor Value",
                      template="plotly_white")
    return fig


def plot_combined_mq135_mq7_chart():
    """Compare MQ135 and MQ7 Predictions"""
    df = fetch_past_data(1000)
    if df is None or 'mq135' not in df.columns or 'mq7' not in df.columns:
        return go.Figure().update_layout(title="Insufficient data for MQ135 and MQ7 combined prediction")

    df = df.dropna(subset=['mq135', 'mq7'])
    df_recent = df.iloc[-24:]
    avg_mq135 = df_recent['mq135'].mean()
    avg_mq7 = df_recent['mq7'].mean()
    future_timestamps = pd.date_range(start=df_recent['created_at'].iloc[-1], periods=24, freq='1H')

    mq135_preds = np.clip(np.random.normal(avg_mq135, 5, 24), 0, None)
    mq7_preds = np.clip(np.random.normal(avg_mq7, 3, 24), 0, None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_timestamps, y=mq135_preds,
                             mode='lines', name='Predicted MQ135', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_timestamps, y=mq7_preds,
                             mode='lines', name='Predicted MQ7', line=dict(color='blue')))
    fig.update_layout(title="Combined MQ135 & MQ7 Prediction",
                      xaxis_title="Time", yaxis_title="Sensor Values",
                      template="plotly_white")
    return fig


def plot_combined_mq135_mq7_dust_chart():
    """Compare MQ135, MQ7 and Dust Predictions"""
    df = fetch_past_data(1000)
    if df is None or 'mq135' not in df.columns or 'mq7' not in df.columns or 'dust' not in df.columns:
        return go.Figure().update_layout(title="Insufficient data for full sensor prediction")

    df = df.dropna(subset=['mq135', 'mq7', 'dust'])
    df_recent = df.iloc[-24:]
    avg_mq135 = df_recent['mq135'].mean()
    avg_mq7 = df_recent['mq7'].mean()
    future_timestamps, _, dust_preds = predict_future_values(df, periods=24)

    mq135_preds = np.clip(np.random.normal(avg_mq135, 5, 24), 0, None)
    mq7_preds = np.clip(np.random.normal(avg_mq7, 3, 24), 0, None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_timestamps, y=mq135_preds,
                             mode='lines', name='Predicted MQ135', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=future_timestamps, y=mq7_preds,
                             mode='lines', name='Predicted MQ7', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=future_timestamps, y=dust_preds,
                             mode='lines', name='Predicted Dust', line=dict(color='green')))
    fig.update_layout(title="Combined MQ135, MQ7 & Dust Prediction",
                      xaxis_title="Time", yaxis_title="Sensor Values",
                      template="plotly_white")
    return fig
