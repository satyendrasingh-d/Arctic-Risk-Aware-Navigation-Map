# Base setup, CSS aur backend logic

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
    # Dummy weather data for 15 features along the route
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

#  Step 2: Sidebar, header aur route input UI
# Sidebar Navigation
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2991/2991231.png", width=50) # Dummy logo
    st.title("Arctic Sentinel AI")
    st.write("AI-Powered Safer Route Intelligence")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "🛣️ Route Analysis", "🗺️ Map View", "⚙️ Model Details", "ℹ️ About"])
    st.markdown("---")
    st.write("👤 **Arctic Sentinel AI**\nv1.0.0")


if page == "🛣️ Route Analysis" or page == "🗺️ Map View":
    st.header("📍 Route Input & Analysis")
    
    # Input Layout
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        start_loc = st.text_input("Start Location", placeholder="e.g., Tromso, Norway")
    with col2:
        end_loc = st.text_input("End Destination", placeholder="e.g., Svalbard")
    with col3:
        st.write("") 
        st.write("")
        analyze_btn = st.button("🔍 Analyze Route")

    # Dhyan rakho: Ye 'if' line theek upar wale 'st.header' ke level par align honi chahiye
    if analyze_btn and start_loc and end_loc:
        with st.spinner("Analyzing AI Risk points..."):
            start_coords = get_coordinates(start_loc)
            end_coords = get_coordinates(end_loc)
            
            if start_coords[0] and end_coords[0]:
                route_coords = get_route(start_coords, end_coords)
                
                if route_coords:
                    # Map Setup
                    m = folium.Map(location=start_coords, zoom_start=5, tiles="CartoDB dark_matter")
                    folium.Marker(start_coords, popup="Start", icon=folium.Icon(color="green")).add_to(m)
                    folium.Marker(end_coords, popup="End", icon=folium.Icon(color="red")).add_to(m)
                    
                    # Generate AI Risk Predictions
                    features = mock_route_features(len(route_coords))
                    if model:
                        probs = model.predict_proba(features)[:, 1]
                    else:
                        probs = np.random.uniform(0, 1, len(route_coords))
                    
                    risk_levels = []
                    colors = []
                    
                    for i, coord in enumerate(route_coords):
                        prob = probs[i]
                        if prob >= 0.65:
                            color, level = '#ff4b4b', 'High'
                        elif prob >= 0.35:
                            color, level = '#ffa500', 'Medium'
                        else:
                            color, level = '#00ff00', 'Low'
                            
                        colors.append(color)
                        risk_levels.append(level)
                        
                        folium.CircleMarker(
                            location=coord, radius=4, color=color, fill=True, fill_color=color, fill_opacity=1
                        ).add_to(m)
                    
                    # Display Layout
                    map_col, stats_col = st.columns([2, 1])
                    
                    with map_col:
                        st.markdown("### 🗺️ Arctic Route Risk Map")
                        st_folium(m, width=700, height=450)
                        
                    with stats_col:
                        st.markdown("### 🛡️ Risk Assessment")
                        avg_risk = np.mean(probs) * 100
                        if avg_risk >= 65:
                            st.error(f"🚨 HIGH RISK ({avg_risk:.1f}%)")
                        elif avg_risk >= 35:
                            st.warning(f"⚠️ MEDIUM RISK ({avg_risk:.1f}%)")
                        else:
                            st.success(f"✅ LOW RISK ({avg_risk:.1f}%)")
                            
                        st.markdown("### 📊 Risk Distribution")
                        df_chart = pd.DataFrame({'Risk': risk_levels})
                        risk_counts = df_chart['Risk'].value_counts().reset_index()
                        risk_counts.columns = ['Risk', 'Count']
                        
                        color_map = {'Low': '#00ff00', 'Medium': '#ffa500', 'High': '#ff4b4b'}
                        fig = px.pie(risk_counts, values='Count', names='Risk', hole=0.7, color='Risk', color_discrete_map=color_map)
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Data Table
                    st.markdown("### 📋 Route Points Analysis")
                    df_table = pd.DataFrame({
                        'Latitude': [c[0] for c in route_coords],
                        'Longitude': [c[1] for c in route_coords],
                        'Risk Level': risk_levels,
                        'Probability (%)': np.round(probs * 100, 2)
                    })
                    st.dataframe(df_table, use_container_width=True, height=250)
                    
            else:
                st.error("Locations not found. Please check city names.")
