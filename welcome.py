import streamlit as st
from streamlit_lottie import st_lottie
import json

# Load Lottie animation
def load_lottiefile(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def show_welcome():
    st.markdown("""
        <style>
        .main-title {
            font-size: 42px;
            font-weight: 800;
            background: linear-gradient(to right, #00C9FF, #92FE9D)import streamlit as st
from streamlit.components.v1 import html
import requests
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from st_pages import show_pages_from_config, add_page_title

st.set_page_config(
    page_title="About Our Project",
    page_icon=":house:",
    layout="wide",
    initial_sidebar_state="auto"
    )

show_pages_from_config()
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("""
          <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Welcome! Pollution Ranger</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background-color: #f3f3f3;
            font-family: 'Arial', sans-serif;
        }

        .text-container {
            text-align: center;
            position: relative;
            top: 50%;
            transform: translateY(-50%);
        }

        .text {
            font-size: 3rem;
            white-space: nowrap;
            overflow: hidden;
            border-right: 0.15em solid;
            animation: animate-text 5s infinite;
            transition: all 0.3s ease;
        }

        .text:hover {
            color: #ff9900;
            transform: scale(1.2);
        }

        @keyframes animate-text {
            0% {
                color: #ff0000;
            }
            10% {
                color: #ff7f00;
            }
            20% {
                color: #ffff00;
            }
            30% {
                color: #00ff00;
            }
            40% {
                color: #00ff7f;
            }
            50% {
                color: #00ffff;
            }
            60% {
                color: #007fff;
            }
            70% {
                color: #0000ff;
            }
            80% {
                color: #7f00ff;
            }
            90% {
                color: #ff00ff;
            }
            100% {
                color: #ff0000;
            }
        }
    </style>
</head>
<body>
    <div class="text-container">
        <h1 class="text">Welcome! Pollution Ranger</h1>
    </div>
</body>
</html>
""", unsafe_allow_html=True
)

from streamlit_lottie import st_lottie

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

about_project = load_lottieurl(r"https://lottie.host/096edf35-97f9-402a-ab07-13438c3190a2/O6iG18SZ9B.json")
features = load_lottieurl("https://lottie.host/6ef5a986-33d7-4dcc-8920-d473e3b2db36/bTb56aUDUm.json")
st.empty()
col1, col2 = st.columns([1.5, 1], gap="small")

with col1:
    st.markdown("<h3 style='text-align: center;color: #008080;'>About Our Project</h3>", unsafe_allow_html=True)
    st.markdown(
    """
    <p style='text-align: center;'>Air pollution is a <span style='color: #008080; font-weight: bold;'>persistent threat</span> in Jakarta, with the air quality consistently at unhealthy levels for the past 5 years.</p>
    <p style='text-align: center;'>Our mission is to <span style='color: #008080; font-weight: bold;'>raise awareness</span> and prompt action. Through our website, we provide real-time air quality data and temperature monitoring, empowering individuals to track their contributions to pollution.</p>
    <p style='text-align: center;'>Recognizing the urgency, we foster a community-driven effort through the <span style='color: #008080; font-weight: bold;'>Pollution Ranger task force</span>. These dedicated individuals monitor and address air pollution within their communities, advocating for change and inspiring others to join the cause.</p>
    """, unsafe_allow_html=True)
with col2:
    st_lottie(about_project, speed=1, reverse=False, loop=True, height=300)

col3,col4 = st.columns([1,3],gap="small")
with col4: 
    st.markdown("<h3 style='text-align: center;color: #4169E1;'>Features</h3>", unsafe_allow_html=True)
    st.markdown(':chart_with_upwards_trend: :blue[Dashboard] : Real-time monitoring of AQI (Air Quality Index) and temperature, along with visualization of pollution conditions in Jakarta.')
    st.markdown(':male-detective: :green[Deep Analysis] : Advanced analysis of pollution data that includes recommendations and ideas for improvement.')
    st.markdown(":ballot_box_with_check: :rainbow[Knowledge Check] : A feature to assess the user's knowledge about air pollution, including statistics (Developer only).")
    st.markdown(":bookmark: :orange[Personal Pollution Tracker] : A calculator to measure or track the amount of pollution or emissions generated in daily activities, including statistics (Developer only).")
    st.caption("*NOTE : It is highly recommended to use a desktop or laptop web browser. However, if you prefer using a smartphone, make sure to enable desktop mode (in Chrome) or use it in landscape orientation.*")
with col3: 
    st_lottie(features, speed=1, reverse=False, loop=True, height=300)

with st.container():
    st.markdown("<h3 style='text-align: center;color: #FF6347;'>Our Team</h3>", unsafe_allow_html=True)
    col5,col6,col7,col8,col9 = st.columns(5, gap="small")
    with col6:
       st.markdown("""
            <style>
                .our-team {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                }

                .profile {
                    margin: 20px;
                    padding: 20px;
                    text-align: center;
                    background-color: #F0FFFF;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    width: 200px;
                    transition: background-color 0.3s;
                }

                .profile:hover {
                    background-color: #AFEEEE;
                }

                .profile img {
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    object-fit: cover;
                    margin-bottom: 15px;
                }

                .profile h3 {
                    font-size: 15px;
                    margin-bottom: 10px;
                }

                .profile a button {
                    background-color: #f08080;
                    color: #fff;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }

                .profile a button:hover {
                    background-color: #2980b9;
                }
            </style>
            """,unsafe_allow_html=True)
       st.markdown("""
            <h3 style='text-align:center;font-size:20px;'> Team Leader </h3>
            <div class="our-team">
                <div class="profile">
                    <img src="https://drive.google.com/uc?export=view&id=1S2YdHLmzTA-m6qCHvbsFeQSfNg1_S7Uc" alt="">
                    <h3 style='color:#183D3D;'>Septian Panjaya</h3>
                    <a href="https://www.linkedin.com/in/septian-panjaya"><button>LinkedIn</button></a>
                </div>
            </div>
            """,unsafe_allow_html=True)
    with col8: 
       st.markdown("""
            <style>
                .our-team {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                }

                .profile {
                    margin: 20px;
                    padding: 20px;
                    text-align: center;
                    background-color: #F0FFFF;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    width: 200px;
                    transition: background-color 0.3s;
                }

                .profile:hover {
                    background-color: #AFEEEE;
                }

                .profile img {
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    object-fit: cover;
                    margin-bottom: 15px;
                }

                .profile h3 {
                    font-size: 15px;
                    margin-bottom: 10px;
                }

                .profile a button {
                    background-color: #f08080;
                    color: #fff;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }

                .profile a button:hover {
                    background-color: #2980b9;
                }
            </style>
            """,unsafe_allow_html=True)
       st.markdown("""
            <h3 style='text-align:center;font-size:20px;'> Team Member </h3>
            <div class="our-team">
                <div class="profile">
                    <img src="https://drive.google.com/uc?export=view&id=1cP4zUoHT6XjW_imoDDFDvnLJn37OKr8o" alt="">
                    <h3 style='color:#183D3D;'>Katon Bagaskara</h3>
                    <a href="https://www.linkedin.com/in/katonbk"><button>LinkedIn</button></a>
                </div>
            </div>
            """,unsafe_allow_html=True)
    col10,col11,col12,col13,col14 = st.columns(5, gap="small")
    with col12:
        st.markdown("""
            <style>
                .our-team {
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: center;
                }

                .profile {
                    margin: 20px;
                    padding: 20px;
                    text-align: center;
                    background-color: #F0FFFF;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    width: 200px;
                    transition: background-color 0.3s;
                }

                .profile:hover {
                    background-color: #AFEEEE;
                }

                .profile img {
                    width: 120px;
                    height: 120px;
                    border-radius: 50%;
                    object-fit: cover;
                    margin-bottom: 15px;
                }

                .profile h3 {
                    font-size: 18px;
                    margin-bottom: 10px;
                }

                .profile a button {
                    background-color: #f08080;
                    color: #fff;
                    border: none;
                    padding: 8px 12px;
                    border-radius: 5px;
                    cursor: pointer;
                    transition: background-color 0.3s;
                }

                .profile a button:hover {
                    background-color: #2980b9;
                }
            </style>
            """,unsafe_allow_html=True)
        st.markdown("""
            <h3 style='text-align:center;font-size:20px;'> Team Coach </h3>
            <div class="our-team">
                <div class="profile">
                    <img src="https://drive.google.com/uc?export=view&id=151O1cYGzt7IavTX8j2r4XlB0Q-g-tkWV" alt="">
                    <h3 style='color:#183D3D;'>I Wayan Nadiantara</h3>
                    <a href="https://www.linkedin.com/in/nadiantara"><button>LinkedIn</button></a>
                </div>
            </div>
            """,unsafe_allow_html=True)

st.markdown(
    """
    <style>
        hr.rainbow {
            height: 1px;
            border: 0;
            background: linear-gradient(to right, violet, indigo, blue, green, yellow, orange, red);
        }
    </style>
    """,
    unsafe_allow_html=True)

st.markdown("<hr class='rainbow'>", unsafe_allow_html=True);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .section {
            font-size: 18px;
            line-height: 1.7;
            color: #E0E0E0;
        }
        .highlight {
            color: #00FFAA;
            font-weight: bold;
        }
        .button-container {
            text-align: center;
            margin-top: 30px;
        }
        .linkedin-btn {
            background-color: #0077B5;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            font-weight: bold;
            border-radius: 30px;
            transition: all 0.3s ease-in-out;
            border: 2px solid #0077B5;
        }
        .linkedin-btn:hover {
            background-color: white;
            color: #0077B5;
            border: 2px solid #0077B5;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">🌫️ Smog Prediction & Air Quality Monitoring</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
            <div class="section">
                📍 <span class="highlight">Location:</span> Faisalabad, Pakistan<br><br>
                ✅ This project provides real-time monitoring of air quality using IoT sensors like <b>MQ135</b>, <b>MQ7</b>, <b>DHT11</b>, and <b>GP2Y1010AU0F</b> integrated with <b>ESP32</b> and <b>ThingSpeak</b>.<br><br>
                📡 Data is uploaded in real-time, analyzed, and forecasted using Machine Learning to assist with smog prediction and public health awareness.<br><br>
                📊 Dashboard offers real-time charts, AQI forecasts, sensor analytics, and weather insights.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="button-container">
                <a class="linkedin-btn" href="https://www.linkedin.com" target="_blank">View on LinkedIn</a>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        lottie_animation = load_lottiefile(r"C:\Users\hp\Desktop\v2\v2\air_quality_dashboard_v2\air_quality_dashboard_v1\assets\Animation - 1747442907291.json")  # ✅ Make sure this file exists
        st_lottie(lottie_animation, speed=1, reverse=False, loop=True, quality="high", height=300)
