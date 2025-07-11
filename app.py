import streamlit as st
import plotly.graph_objs as go
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import os

# Import custom modules
from utils import fetch_latest_data, get_aqi_category, calculate_data_freshness
from prediction import (
    plot_prediction_chart,
    plot_aqi_distribution_forecast,
    plot_hourly_forecast,
    plot_sensor_correlations,
    generate_forecast_metrics,
    plot_comparison_chart,
    plot_mq135_prediction_chart,
    plot_mq7_prediction_chart,
    plot_combined_mq135_mq7_chart,
    plot_combined_mq135_mq7_dust_chart,
    plot_mq135_dust_combined_chart
)
from weather import WeatherService

# Page configuration
st.set_page_config(
    page_title="Smog Prediction Dashboard",
    page_icon="🌫️",
    layout="wide"
)


def display_home_page():
    """Display the home page content."""
    # Modern header with animation effect
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #1e3c72, #2a5298); 
         color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h1 style='margin:0; font-size: 2.5rem;'>🌫️ Smog Prediction & Air quality Monitoring</h1>
        <p style='margin:5px 0 0; font-size: 1.2rem;'>Raheem Gareden, Faisalabad, Pakistan</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Spacer
    st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
    
    # Fetch latest data
    latest_data = fetch_latest_data()
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)

    if latest_data:
        dust_value = latest_data.get('dust', 0)
        temp_value = latest_data.get('temperature', 0)
        humidity_value = latest_data.get('humidity', 0)
        co_value = latest_data.get('co', 0)
        aqi_info = get_aqi_category(dust_value)
        freshness, color = calculate_data_freshness(latest_data['timestamp'])
        
        # Column 1: AQI with visual indicator
        with col1:
            text_color = 'white' if aqi_info.get('category') in ['Unhealthy', 'Very Unhealthy', 'Hazardous'] else 'black'
            st.markdown(f"""
            <div style='background-color: {aqi_info.get('color', '#FFFFFF')}; padding: 15px; 
                 border-radius: 10px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <h3 style='margin:0; color: {text_color}'>Current Air Quality</h3>
                <div style='font-size: 2rem; font-weight: bold; margin: 10px 0; color: {text_color}'>
                    {aqi_info.get('category', 'Unknown')}
                </div>
                <div style='color: {text_color}'>{dust_value} µg/m³</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Column 2: Temperature and Humidity with icons
        with col2:
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
                 box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                <div style='display: flex; align-items: center;'>
                    <div style='font-size: 2rem; margin-right: 15px;'>🌡️</div>
                    <div>
                        <h3 style='margin: 0;'>Temperature</h3>
                        <div style='font-size: 1.5rem; font-weight: bold;'>{temp_value}°C</div>
                    </div>
                </div>
                <div style='height: 10px'></div>
                <div style='display: flex; align-items: center;'>
                    <div style='font-size: 2rem; margin-right: 15px;'>💧</div>
                    <div>
                        <h3 style='margin: 0;'>Humidity</h3>
                        <div style='font-size: 1.5rem; font-weight: bold;'>{humidity_value}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Column 3: MQ-135 Air Quality Score with D-type gauge chart
        with col3:
            mq135_value = latest_data.get('mq135', 0)
            # Convert MQ135 value to a normalized air quality score (0-100)
            # Assuming MQ135 values typically range from 0-1000, adjust as needed
            air_quality_score = min(100, max(0, (mq135_value / 10)))
            
            # Determine quality category and color based on the score
            if air_quality_score < 20:
                quality_category = "Excellent"
                gauge_color = "#00CC00"  # Green
            elif air_quality_score < 40:
                quality_category = "Good"
                gauge_color = "#AACC00"  # Light Green
            elif air_quality_score < 60:
                quality_category = "Moderate"
                gauge_color = "#FFFF00"  # Yellow
            elif air_quality_score < 80:
                quality_category = "Poor"
                gauge_color = "#FF9900"  # Orange
            else:
                quality_category = "Hazardous"
                gauge_color = "#FF0000"  # Red
            
            # Create a D-type gauge chart using Plotly
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=air_quality_score,
                title={'text': "Air Quality (MQ-135)", 'font': {'size': 24}},
                delta={'reference': 50, 'increasing': {'color': "#FF0000"}, 'decreasing': {'color': "#00CC00"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': gauge_color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 20], 'color': '#00CC00'},  # Green
                        {'range': [20, 40], 'color': '#AACC00'},  # Light Green
                        {'range': [40, 60], 'color': '#FFFF00'},  # Yellow
                        {'range': [60, 80], 'color': '#FF9900'},  # Orange
                        {'range': [80, 100], 'color': '#FF0000'}  # Red
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            # Update the layout of the gauge chart
            fig.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#404040", 'family': "Arial"}
            )
            
            # Display the gauge chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Display the quality category and freshness info
            st.markdown(f"""
            <div style='background-color: {gauge_color}; padding: 10px; border-radius: 10px; 
                 box-shadow: 0 4px 6px rgba(0,0,0,0.1); text-align: center; margin-top: -20px;'>
                <div style='font-size: 1.2rem; font-weight: bold; color: {"white" if air_quality_score > 60 else "black"};'>
                    {quality_category} ({mq135_value} ppm)
                </div>
            </div>
            
            <div style='display: flex; align-items: center; margin-top: 10px;'>
                <div style='font-size: 1.5rem; margin-right: 10px;'>⏱️</div>
                <div>
                    <div style='font-size: 0.9rem;'>Data Freshness</div>
                    <div style='font-weight: bold; color: {color};'>{freshness}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        # Show a placeholder if data is not available
        with col1:
            st.warning("No real-time data available. Please check your sensor connection.")
    
    # Spacer
    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
    
    # Information Section Header
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
         box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;'>
        <h2 style='margin-top: 0;'>🔍 About This Dashboard</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for information sections
    display_info_tabs()
    
    # Spacer
    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
    
    # Visual section with AQI reference and live map
    viz_col1, viz_col2 = st.columns([1, 1])
    
    with viz_col1:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
             box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2 style='margin-top: 0;'>📷 AQI Levels Reference</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.image("assets/2236_WHO_Guidlines_Chart_UPDATE_AQI_2024.webp", caption="Air Quality Index Categories", use_container_width=True)
    
    with viz_col2:
        st.markdown("""
        <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; 
             box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <h2 style='margin-top: 0;'>📍 Monitoring Station Location</h2>
        </div>
        """, unsafe_allow_html=True)
        
        display_station_map()
    
    # Spacer
    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
    
    # Call-to-action section
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1e3c72, #2a5298); padding: 20px; 
         border-radius: 10px; color: white; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
        <h2 style='margin-top: 0;'>🔔 Stay Informed About Air Quality</h2>
        <p style='font-size: 1.2rem;'>Access real-time data, historical trends, and forecasts using the tabs above.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Spacer and footer
    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 10px; color: #6c757d; font-size: 0.8rem;'>
        Air Quality Monitoring Dashboard • Powered by Streamlit • © 2025
    </div>
    """, unsafe_allow_html=True)


def display_info_tabs():
    """Display information tabs about the dashboard."""
    info_tab1, info_tab2, info_tab3 = st.tabs(["📊 System Overview", "🔬 Monitoring Network", "ℹ️ How It Works"])
    
    with info_tab1:
        st.markdown("""
        This dashboard provides real-time air quality monitoring for Faisalabad, Pakistan, 
        with a focus on smog prediction and pollution tracking.
        
        ### Key Features:
        - Real-time monitoring of PM2.5, CO, temperature, and humidity
        - Historical data analysis and trend visualization
        - Pollution forecasting using machine learning
        - Health advisories based on air quality levels
        
        ### Data Sources:
        - **Sensors:** MQ135, MQ7, DHT11, PM2.5 Dust Sensor
        - **Weather Data:** OpenWeatherMap API
        - **Data Storage:** Cloud-based analytics platform
        - **Visualization:** Streamlit for interactive dashboards
        - **Forecasting:** Machine learning models for predictive analytics
        """)
    
    with info_tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🛰️ Hardware Infrastructure
            - **Main Sensors:** 
              - MQ7 (Carbon Monoxide)
              - MQ135 (Gases & Air Quality)
              - DHT11 (Temperature & Humidity)
              - PM2.5 Dust Sensor
            - **Microcontroller:** ESP32 with WiFi
            - **Power:** Solar with battery backup
            - **Housing:** Weather-resistant enclosure
            """)
            
        with col2:
            st.markdown("""
            ### 🌐 Network Architecture
            - **Data Transmission:** MQTT Protocol
            - **Cloud Storage:** Time-series database
            - **Processing:** Real-time & batch analytics
            - **Redundancy:** Multiple measurement points
            - **Uptime:** 24/7 monitoring with failover
            """)
    
    with info_tab3:
        st.markdown("""
        ### How Our System Works
        
        1. **Data Collection:** Our IoT devices continuously measure air quality parameters
        2. **Transmission:** Data is sent over secure WiFi connections to our cloud platform
        3. **Processing:** Raw data is cleaned, calibrated, and converted to standard units
        4. **Analysis:** Machine learning models detect anomalies and predict trends
        5. **Visualization:** Dashboard updates in real-time with insights and alerts
        
        The system maintains historical records for research purposes and provides public health 
        recommendations based on current conditions.
        """)


def display_station_map():
    """Display a map showing the monitoring station location."""
    m = folium.Map(location=[31.373473, 73.056772], zoom_start=12, tiles="CartoDB positron")
    
    # Add a custom icon for the monitoring station
    icon = folium.features.CustomIcon("assets/sensor-icon.png", icon_size=(30, 30)) if os.path.exists("assets/sensor-icon.png") else folium.Icon(color='red', icon='cloud')
    
    # Add a circle to show coverage area
    folium.Circle(
        location=[31.373473, 73.056772],
        radius=1000,  # 2 km radius
        color='#1e3c72',
        fill=True,
        fill_opacity=0.1
    ).add_to(m)
    
    # Add marker for the monitoring station
    folium.Marker(
        [31.373473, 73.056772],
        popup="<b>Air Quality Monitoring Station</b><br>Real-time PM2.5, CO, Temperature & Humidity",
        tooltip="Smog Monitoring Node",
        icon=icon
    ).add_to(m)
    
    # Show the map
    st_folium(m, width=None, height=400)


def display_real_time_data():
    """Display real-time sensor readings."""
    st.header("📊 Real-Time Sensor Readings")
    latest_data = fetch_latest_data()
    
    if latest_data:
        freshness_text, freshness_color = calculate_data_freshness(latest_data['timestamp'])
        st.markdown(f"🕒 **Data Freshness:** <span style='color:{freshness_color}'>{freshness_text}</span>", 
                    unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Temperature", f"{latest_data.get('temperature', 'N/A')}°C")
            st.metric("Humidity", f"{latest_data.get('humidity', 'N/A')}%")
        with col2:
            st.metric("MQ135 (Air Quality)", f"{latest_data.get('mq135', 0)}")
            st.metric("MQ7 (CO Levels)", f"{latest_data.get('mq7', 0)}")
        with col3:
            dust_value = latest_data.get('dust', 0)
            aqi_info = get_aqi_category(dust_value)
            st.metric("Dust Level (PM2.5)", f"{dust_value} µg/m³", delta=aqi_info.get('category', 'Unknown'))

        st.subheader("🩺 Health Advisory")
        st.warning(aqi_info.get('description', 'No specific advice available.'))
    else:
        st.error("Unable to fetch real-time data. Please check sensor connections.")


def display_predictions():
    """Display prediction charts and forecasts."""
    st.header("🔮 Air Quality Forecasting")

    with st.expander("🧾 Graph Summary Table"):
        summary_df = pd.DataFrame([
            ["Dust Level Trend Prediction", "Dust", "Time Series Forecast", 
             "Predicts future dust levels and air pollution trend for 24 hours."],
            ["24-Hour AQI Distribution", "Dust", "Category Distribution", 
             "Shows the proportion of air quality categories in next 24 hours."],
            ["Hourly PM2.5 Forecast", "Dust", "Hourly Trend", 
             "Visualizes hourly air pollution levels to plan short-term activities."],
            ["Past vs Future AQI", "Dust", "Comparison Analysis", 
             "Compares past 24 hours with forecasted 24 hours of AQI."],
            ["MQ135 Forecast", "MQ135", "Sensor Forecast", 
             "Predicts future gas concentration trends measured by MQ135 sensor."],
            ["MQ7 Forecast", "MQ7", "Sensor Forecast", 
             "Forecasts CO levels and helps detect dangerous exposure periods."],
            ["MQ135 + Dust Forecast", "MQ135, Dust", "Correlation Forecast", 
             "Shows combined behavior of dust and air quality gases."],
            ["MQ135 + MQ7 Forecast", "MQ135, MQ7", "Multi-Gas Prediction", 
             "Detects combined gas pollution trends for general air quality."],
            ["MQ135 + MQ7 + Dust Forecast", "MQ135, MQ7, Dust", "Multi-Factor Forecast", 
             "Complete overview of air quality using all key sensor inputs."]
        ], columns=["Graph Name", "Sensor(s) Involved", "Insight Type", "Details"])
        st.dataframe(summary_df, use_container_width=True)

    # Main prediction tabs
    pred_tab1, pred_tab2, pred_tab3 = st.tabs(["Trend Prediction", "AQI Distribution", "Hourly Forecast"])
    
    with pred_tab1:
        st.subheader("Dust Level Trend Prediction")
        st.plotly_chart(plot_prediction_chart(), use_container_width=True)

    with pred_tab2:
        st.subheader("24-Hour AQI Distribution")
        st.plotly_chart(plot_aqi_distribution_forecast(), use_container_width=True)

    with pred_tab3:
        st.subheader("Hourly PM2.5 Forecast")
        st.plotly_chart(plot_hourly_forecast(), use_container_width=True)

    # Forecast metrics
    st.subheader("📊 Forecast Summary")
    forecast_metrics = generate_forecast_metrics()
    if forecast_metrics['status'] == 'success':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current AQI", f"{forecast_metrics['current']['value']} µg/m³", 
                     delta=forecast_metrics['current']['category'])
        with col2:
            st.metric("24hr Avg Forecast", f"{forecast_metrics['forecast']['average']} µg/m³", 
                     delta=forecast_metrics['forecast']['dominant_category']['category'])
        with col3:
            st.metric("Peak Forecast", f"{forecast_metrics['forecast']['max']['value']} µg/m³", 
                     delta=f"At {forecast_metrics['forecast']['max']['time']}")

    # Additional charts in expandable sections
    with st.expander("📉 Compare Past vs Future AQI"):
        st.plotly_chart(plot_comparison_chart(), use_container_width=True)
    with st.expander("📉 MQ135 Forecast"):
        st.plotly_chart(plot_mq135_prediction_chart(), use_container_width=True)
    with st.expander("📉 MQ7 Forecast"):
        st.plotly_chart(plot_mq7_prediction_chart(), use_container_width=True)
    with st.expander("📉 Combined MQ135 & MQ7 Forecast"):
        st.plotly_chart(plot_combined_mq135_mq7_chart(), use_container_width=True)
    with st.expander("📉 Combined MQ135, MQ7 & Dust Forecast"):
        st.plotly_chart(plot_combined_mq135_mq7_dust_chart(), use_container_width=True)


def display_weather():
    """Display weather information and forecasts."""
    st.header("🌡️ Weather Insights")
    weather_service = WeatherService()
    weather_data = weather_service.get_weather_data()
    
    if weather_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Temperature", f"{weather_data['current']['temp']}°C")
            st.metric("Feels Like", f"{weather_data['current']['feels_like']}°C")
        with col2:
            st.metric("Humidity", f"{weather_data['current']['humidity']}%")
            st.metric("Wind Speed", f"{weather_data['wind']['speed']} m/s")
        with col3:
            st.metric("Condition", weather_data['current']['condition'])
            st.metric("Wind Direction", weather_data['wind']['direction'])
            st.image(weather_data['current']['icon'], width=80, caption="Weather Condition")
        
        st.subheader("🌈 Hourly Temperature Forecast")
        st.plotly_chart(weather_service.plot_hourly_temperature_forecast(), use_container_width=True)
    else:
        st.error("Unable to fetch weather data.")


def display_advanced_analytics():
    """Display advanced analytics and sensor correlations."""
    st.header("📈 Sensor Correlation & Advanced Insights")
    st.subheader("🔬 Sensor Correlation Analysis")
    st.plotly_chart(plot_sensor_correlations(), use_container_width=True)

    st.subheader("🧠 Insights & Observations")
    latest_data = fetch_latest_data()
    if latest_data:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Highest Correlated Sensors", "Temperature & Dust")
        with col2:
            st.metric("Potential Health Risk", "Moderate")
        with col3:
            st.metric("Recommendation", "Monitor Outdoor Activities")


def display_station_info():
    """Display monitoring station information."""
    st.header("📍 Monitoring Station Details")
    st.markdown("""
        ### Air Quality Monitoring Station

        **Location:** Raheem Garden Faisalabad, Punjab, Pakistan  

        #### 🛠 Sensor Equipment
        - **Microcontroller:** ESP32
        - **Air Quality Sensors:** MQ135, MQ7
        - **Environmental Sensors:** DHT11, Dust Sensor

        #### 📡 Data Transmission
        - **Platform:** ThingSpeak IoT
        - **Update Frequency:** Real-time
        - **Data Storage:** Cloud-based Analytics
    """)

    st.subheader("⚙️ Technical Configuration")
    config_data = {
        "Sampling Rate": "Every 5 minutes",
        "Sensor Calibration": "Periodic",
        "Data Transmission Protocol": "MQTT",
        "Power Source": "5V DC",
        "Data Storage": "Cloud (ThingSpeak)",
        "Data Format": "csv",
    }
    config_df = pd.DataFrame.from_dict(config_data, orient='index', columns=['Value'])
    st.table(config_df)


def load_sidebar_forecast_summary():
    """Load and display forecast summary in sidebar."""
    try:
        summary = generate_forecast_metrics()
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"📢 **Forecast**: {summary['forecast']['dominant_category']['category']}")
        st.sidebar.markdown(f"📈 **Trend**: {summary['trend']['emoji']} {summary['trend']['direction']}")
    except:
        st.sidebar.markdown("⚠️ Forecast data unavailable.")


def main():
    """Main function to run the application."""
    st.sidebar.title("🌫️ Air Quality Dashboard")
    selected_tab = st.sidebar.radio(
        "Navigation",
        ["🏠 Home", "📊 Real-Time Data", "🔮 Predictions", "🌡️ Weather", 
         "📈 Advanced Analytics", "📍 Station Info"]
    )

    # Load forecast summary in sidebar
    load_sidebar_forecast_summary()

    # Display selected page
    if selected_tab == "🏠 Home":
        display_home_page()
    elif selected_tab == "📊 Real-Time Data":
        display_real_time_data()
    elif selected_tab == "🔮 Predictions":
        display_predictions()
    elif selected_tab == "🌡️ Weather":
        display_weather()
    elif selected_tab == "📈 Advanced Analytics":
        display_advanced_analytics()
    elif selected_tab == "📍 Station Info":
        display_station_info()


if __name__ == "__main__":
    main()