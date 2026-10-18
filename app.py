import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import requests
import numpy as np
import joblib
import plotly.express as px

# Page configuration
st.set_page_config(page_title="Arctic Sentinel AI", layout="wide", initial_sidebar_state="expanded")

# Dark Theme Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #0b1426; color: white; }
    .css-1d391kg, .css-12oz5g7 { background-color: #111c33; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #ffffff; }
    .stTextInput>div>div>input { background-color: #1c2a47; color: white; border-radius: 5px; }
    .stButton>button { background-color: #0d6efd; color: white; border-radius: 5px; width: 100%; }
    .stDataFrame { background-color: #111c33; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        return joblib.load('arctic_master_final_model_v3.pkl')
    except Exception as e:
        return None

model = load_model()

def get_coordinates(city_name):
    url = f"https://nominatim.openstreetmap.org/search?q={city_name}&format=json&limit=1"
    headers = {'User-Agent': 'ArcticRouteApp/1.0'}
    response = requests.get(url, headers=headers).json()
    if response:
        return float(response[0]['lat']), float(response[0]['lon'])
    return None, None

def get_route(start_coords, end_coords):
    lon1, lat1 = start_coords[1], start_coords[0]
    lon2, lat2 = end_coords[1], end_coords[0]
    url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
    response = requests.get(url).json()
    if response.get("code") == "Ok":
        route_coords = response["routes"][0]["geometry"]["coordinates"]
        return [[coord[1], coord[0]] for coord in route_coords] 
    return None

def get_live_weather_features(route_coords, api_key):
    if len(route_coords) > 10:
        indices = np.linspace(0, len(route_coords)-1, 10, dtype=int)
        sampled_coords = [route_coords[i] for i in indices]
    else:
        sampled_coords = route_coords
        
    features_list = []
    
    for lon, lat in sampled_coords:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        try:
            resp = requests.get(url, timeout=5).json()
            if resp.get("cod") == 200:
                temp = resp['main']['temp']
                wind_kph = resp['wind']['speed'] * 3.6
                precip = resp.get('rain', {}).get('1h', 0)
                snow = resp.get('snow', {}).get('1h', 0)
                visibility = resp.get('visibility', 10000)
                pressure = resp['main']['pressure']
                
                is_freezing = 1 if temp <= 0 else 0
                black_ice = 1 if temp <= 2 and (precip > 0 or snow > 0) else 0
                severe = 1 if wind_kph > 50 or snow > 2 else 0
                
                features_list.append({
                    'temp': temp, 'snowfall': snow, 'precip': precip, 'wind': wind_kph,
                    'Wind_Chill': temp - 2 if wind_kph > 10 else temp,
                    'Is_Freezing': is_freezing, 'Severe_Weather': severe,
                    'Risk_Zone': 2 if severe else (1 if is_freezing else 0),
                    'Zone_Weather_Risk': wind_kph / 10, 'Is_Night': 0, 'Is_RushHour': 0,
                    'Visibility_m': visibility, 'Snow_Depth_m': snow / 10,
                    'Surface_Pressure_hPa': pressure, 'Black_Ice_Risk': black_ice
                })
            else:
                continue
        except Exception as e:
            continue
            
    if not features_list:
        return None, None
    return pd.DataFrame(features_list), sampled_coords

if 'analyzed' not in st.session_state: st.session_state['analyzed'] = False
if 'route_coords' not in st.session_state: st.session_state['route_coords'] = None
if 'start_coords' not in st.session_state: st.session_state['start_coords'] = None
if 'end_coords' not in st.session_state: st.session_state['end_coords'] = None
if 'live_route_points' not in st.session_state: st.session_state['live_route_points'] = None
if 'live_probs' not in st.session_state: st.session_state['live_probs'] = None
with st.sidebar:
    st.title("Arctic Sentinel AI")
    page = st.radio("Navigation", ["🏠 Home", "🛣️ Route Analysis", "🗺️ Map View", "⚙️ Model Details", "ℹ️ About"])

if page == "🏠 Home":
    st.title("Welcome to Arctic Sentinel AI")
    st.write("AI-Powered Safer Route Intelligence dashboard with Live OpenWeather API.")

elif page == "⚙️ Model Details":
    st.title("Model Details")
    st.write("Model: Stacking Classifier V3 | API: OpenWeatherMap Live Data")

elif page == "ℹ️ About":
    st.title("About")
    st.write("Developed by Satyendra. Admin panel will be added in future updates.")

elif page == "🛣️ Route Analysis":
    st.header("📍 Route Input & Analytics")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: start_loc = st.text_input("Start Location", placeholder="e.g., Yakutsk, Russia")
    with col2: end_loc = st.text_input("End Destination", placeholder="e.g., Magadan, Russia")
    with col3: 
        st.write(""); st.write("")
        if st.button("🔍 Analyze Route"):
            # YAHAN APNI ASLI API KEY DAALO
            API_KEY = "17d1849df01bb168a74f963e12235c00" 
            
            if API_KEY == "APi_KEY":
                st.error("Please enter your OpenWeather API Key in the code first!")
            else:
                with st.spinner("Fetching route and live weather data..."):
                    sc = get_coordinates(start_loc)
                    ec = get_coordinates(end_loc)
                    if sc[0] and ec[0]:
                        full_route = get_route(sc, ec)
                        if full_route:
                            features, sampled_route = get_live_weather_features(full_route, API_KEY)
                            if features is not None:
                                probs = model.predict_proba(features)[:, 1] if model else np.random.uniform(0, 1, len(sampled_route))
                                st.session_state.update({
                                    'start_coords': sc, 'end_coords': ec, 'route_coords': full_route,
                                    'live_route_points': sampled_route, 'live_probs': probs, 'analyzed': True
                                })
                            else:
                                st.error("Failed to fetch weather data.")
                        else:
                            st.error("Could not find a driving route between these locations.")
                    else:
                        st.error("Locations not found.")
            
    if st.session_state.get('analyzed') and st.session_state.get('live_probs') is not None:
        avg_risk = np.mean(st.session_state['live_probs']) * 100
        st.markdown("### 🛡️ Real-Time Risk Assessment")
        if avg_risk >= 65: st.error(f"🚨 HIGH RISK ({avg_risk:.1f}%) - Extreme Weather Detected!")
        elif avg_risk >= 35: st.warning(f"⚠️ MEDIUM RISK ({avg_risk:.1f}%) - Drive with Caution.")
        else: st.success(f"✅ LOW RISK ({avg_risk:.1f}%) - Route is Clear.")

elif page == "🗺️ Map View":
    st.header("🗺️ Live Route Map")
    if st.session_state['analyzed'] and st.session_state['start_coords']:
        m = folium.Map(location=st.session_state['start_coords'], zoom_start=5, tiles="OpenStreetMap") 
        
        # Draw full route path
        folium.PolyLine(locations=st.session_state['route_coords'], color='blue', weight=3).add_to(m)
        
        # Draw risk points based on live weather
        for idx, coord in enumerate(st.session_state['live_route_points']):
            prob = st.session_state['live_probs'][idx]
            color = "red" if prob >= 0.65 else ("orange" if prob >= 0.35 else "green")
            folium.CircleMarker(location=coord, radius=6, color=color, fill=True, fill_opacity=0.9).add_to(m)
            
        folium.Marker(st.session_state['start_coords'], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(st.session_state['end_coords'], popup="End", icon=folium.Icon(color="red")).add_to(m)
        
        st_folium(m, width=800, height=500)
    else:
        st.warning("Pehle 'Route Analysis' me jakar route search karo.")
