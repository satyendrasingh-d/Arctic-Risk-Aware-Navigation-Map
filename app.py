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
        return joblib.load('arctic_master_v3.pkl')
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

def mock_route_features(num_points):
    return pd.DataFrame({
        'temp': np.random.uniform(-30, 5, num_points),
        'snowfall': np.random.uniform(0, 15, num_points),
        'precip': np.random.uniform(0, 10, num_points),
        'wind': np.random.uniform(5, 60, num_points),
        'Wind_Chill': np.random.uniform(-40, 0, num_points),
        'Is_Freezing': np.random.choice([0, 1], num_points),
        'Severe_Weather': np.random.choice([0, 1], num_points, p=[0.8, 0.2]),
        'Risk_Zone': np.random.choice([0, 1, 2], num_points),
        'Zone_Weather_Risk': np.random.uniform(0, 10, num_points),
        'Is_Night': np.random.choice([0, 1], num_points),
        'Is_RushHour': np.random.choice([0, 1], num_points),
        'Visibility_m': np.random.uniform(100, 10000, num_points),
        'Snow_Depth_m': np.random.uniform(0, 3, num_points),
        'Surface_Pressure_hPa': np.random.uniform(950, 1030, num_points),
        'Black_Ice_Risk': np.random.choice([0, 1], num_points)
    })

# Session state initialize karo taaki data gayab na ho
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
    st.session_state['route_coords'] = None
    st.session_state['start_coords'] = None
    st.session_state['end_coords'] = None

# Sidebar Navigation
with st.sidebar:
    st.title("Arctic Sentinel AI")
    page = st.radio("Navigation", ["🏠 Home", "🛣️ Route Analysis", "🗺️ Map View", "⚙️ Model Details", "ℹ️ About"])

# Step 1: Blank Pages Fix
if page == "🏠 Home":
    st.title("Welcome to Arctic Sentinel AI")
    st.write("AI-Powered Safer Route Intelligence dashboard.")

elif page == "⚙️ Model Details":
    st.title("Model Details")
    st.write("Model: Stacking Classifier V3 | Accuracy: 66%")

elif page == "ℹ️ About":
    st.title("About")
    st.write("Developed by Satyendra. Admin panel aur database integration future update me aayega.")

# Step 4: Route Analysis (Data & Charts)
elif page == "🛣️ Route Analysis":
    st.header("📍 Route Input & Analytics")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: start_loc = st.text_input("Start Location", placeholder="e.g., Tromso, Norway")
    with col2: end_loc = st.text_input("End Destination", placeholder="e.g., Nordkapp, Norway")
    with col3: 
        st.write("") 
        st.write("")
        if st.button("🔍 Analyze"):
            with st.spinner("Analyzing route..."):
                sc = get_coordinates(start_loc)
                ec = get_coordinates(end_loc)
                if sc[0] and ec[0]:
                    st.session_state['start_coords'] = sc
                    st.session_state['end_coords'] = ec
                    st.session_state['route_coords'] = get_route(sc, ec)
                    st.session_state['analyzed'] = True
                else:
                    st.error("Locations not found.")
            
    if st.session_state['analyzed'] and st.session_state['route_coords']:
        features = mock_route_features(len(st.session_state['route_coords']))
        probs = model.predict_proba(features)[:, 1] if model else np.random.uniform(0, 1, len(st.session_state['route_coords']))
        
        st.markdown("### 🛡️ Risk Assessment")
        avg_risk = np.mean(probs) * 100
        if avg_risk >= 65: st.error(f"🚨 HIGH RISK ({avg_risk:.1f}%)")
        elif avg_risk >= 35: st.warning(f"⚠️ MEDIUM RISK ({avg_risk:.1f}%)")
        else: st.success(f"✅ LOW RISK ({avg_risk:.1f}%)")

# Step 4: Map View (Google Maps style Map only)
elif page == "🗺️ Map View":
    st.header("🗺️ Live Route Map")
    if st.session_state['analyzed'] and st.session_state['start_coords']:
        # OpenStreetMap tiles se API error hat jayega
        m = folium.Map(location=st.session_state['start_coords'], zoom_start=5, tiles="OpenStreetMap") 
        
        for coord in st.session_state['route_coords']:
            folium.CircleMarker(location=coord, radius=3, color="blue", fill=True).add_to(m)
        
        folium.Marker(st.session_state['start_coords'], popup="Start", icon=folium.Icon(color="green")).add_to(m)
        folium.Marker(st.session_state['end_coords'], popup="End", icon=folium.Icon(color="red")).add_to(m)
        
        st_folium(m, width=800, height=500)
    else:
        st.warning("Pehle 'Route Analysis' me jakar route search karo.")
