# 🌫️ Smog Prediction & Air Quality Monitoring Dashboard

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red)

A full-stack Streamlit dashboard for real-time air quality monitoring and smog prediction, powered by ESP32 sensor data and a Random Forest machine learning model. Developed as a Final Year Project to provide actionable insights into air pollution trends.

---

## 🚀 Features

- 📡 **Real-time Data**: Fetches air quality metrics from **ThingSpeak** using ESP32 sensors.
- 🧠 **Smog Prediction**: Uses a trained **RandomForestRegressor** to forecast PM2.5 levels.
- 🌡️ **Multi-Sensor Visualization**: Displays Temperature, Humidity, Dust (PM2.5), CO, and CO₂.
- 🗺️ **Health Insights**: Offers weather-based health tips and smog prevention advice.
- 📊 **Interactive Dashboard**: Built with **Streamlit** for intuitive data exploration.
- 🔮 **AQI Forecasting**: Predicts future Air Quality Index (AQI) based on historical trends.

---

## 🧰 Tech Stack

| Layer              | Technologies Used                            |
|--------------------|---------------------------------------------|
| **Hardware**       | ESP32, MQ135, MQ7, GP2Y1010AU0F, DHT11      |
| **Data Handling**  | ThingSpeak API, CSV                         |
| **Machine Learning**| Scikit-learn (Random Forest)                |
| **Frontend**       | Streamlit                                   |
| **Visualization**  | Matplotlib, Seaborn                         |
| **Backend**        | Python (Pandas, NumPy, Requests)            |

---

## 🔌 Sensor Setup

| Sensor        | Metric                  | GPIO Pin       |
|---------------|-------------------------|----------------|
| **DHT11**     | Temperature & Humidity  | GPIO 4         |
| **MQ135**     | CO₂ levels             | GPIO 34        |
| **MQ7**       | CO levels              | GPIO 35        |
| **Dust Sensor**| PM2.5                  | GPIO 32 & 25   |

Data is transmitted to [ThingSpeak Channel](https://thingspeak.com/channels/2936691).

---

## 📊 Machine Learning Model

- **Input Features**: Temperature, Humidity, CO, CO₂, PM2.5
- **Algorithm**: RandomForestRegressor
- **Purpose**: Predict PM2.5 concentrations for smog risk forecasting
- **Training Data**: Collected via ESP32, stored in ThingSpeak, and exported as CSV
- **Performance**: Evaluated using RMSE and R² metrics

---

## 🖼️ Dashboard Preview

![Dashboard Home](assets/dashboard_home.png)
![Prediction Page](assets/prediction_page.png)

---

## 🛠️ Installation Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/rizwan-muhammad-ai/Air-Qaulity-Monitoring-dashboard.git
   cd smog-prediction-dashboard
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Dashboard**
   ```bash
   streamlit run app.py
   ```

---

## 🤖 Usage Instructions

1. **View Real-time Data**: Monitor sensor readings from ESP32 via ThingSpeak.
2. **Predict Smog Levels**: Get PM2.5 predictions based on live and historical data.
3. **Explore Trends**: Analyze AQI trends and pollutant correlations.
4. **Access Health Tips**: Read personalized advice for smog prevention.

---

## ✅ Project Status

- ✅ Sensor data collection via ESP32 and ThingSpeak
- ✅ Random Forest model trained and integrated
- ✅ Streamlit dashboard with 4 interactive pages
- 🚧 **Future Work**:
  - Enhance model accuracy with additional data
  - Implement AQI categorization and alert system
  - Integrate time-series forecasting for long-term predictions

---

## 👨‍💻 Author

**Muhammad Rizwan**  
📫 Email: rizwan.m5414@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/)  


---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---
