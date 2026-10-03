import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os
import folium
from streamlit_folium import st_folium
# 2. PAGE CONFIGURATION
st.set_page_config(
    page_title="Arctic SafeRoute AI",
    page_icon="🧊",
    layout="wide")

# 3. FILE PATHS
MODEL_PATH = "random_forest_final.pkl"
PREDICTION_PATH = "final_deployment_predictions.csv"

# 4. MODEL LOADING
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as file:
        return pickle.load(file)
try:
    model = load_model()
    st.success("Random Forest model loaded successfully.")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()
  
# 5. DEPLOYMENT DATA LOADING
try:
    deployment_data = pd.read_csv(PREDICTION_PATH)
    st.success("Deployment prediction data loaded successfully.")
except Exception as e:
    st.error(f"Prediction data loading error: {e}")
    st.stop()
  
# 6. MAIN APPLICATION HEADER
st.title("🧊 Arctic SafeRoute AI")
st.subheader("AI-Based Arctic Route Risk Assessment")

# 7. ROUTE RISK SUMMARY
st.markdown("### Route Risk Overview")
st.write(
    "The system evaluates Arctic route points and "
    "identifies their predicted risk levels.")

# 8. INTERACTIVE ARCTIC MAP
st.markdown("### Arctic Route Risk Map")
# Map code will be added here.

# 9. ALTERNATIVE ROUTE COMPARISON
st.markdown("### Alternative Route Options")
# Route comparison code will be added here.

# 10. USER ROUTE SELECTION
st.markdown("### Choose Your Route")
# User route selection code will be added here.

# 11. FINAL ROUTE DECISION
st.markdown("### Selected Route")
# Final route display code will be added here.

# ============================================================
# 8. INTERACTIVE ARCTIC MAP
# ============================================================
st.markdown("### Arctic Route Risk Map")
try:
    import folium
    from streamlit_folium import st_folium
    required_map_columns = [
        "latitude",
        "longitude",
        "rf_risk_prediction",
        "rf_risk_probability",
        "rf_risk_level"]
    missing_map_columns = [
        column for column in required_map_columns
        if column not in deployment_data.columns
    ]
    if missing_map_columns:
        st.error(
            f"Map data missing columns: {missing_map_columns}")
        st.stop()
    map_data = deployment_data.dropna(
        subset=["latitude","longitude","rf_risk_level"]
    ).copy()
    arctic_map = folium.Map(
        location=[
            map_data["latitude"].mean(),
            map_data["longitude"].mean()
        ],
        zoom_start=8,
        tiles="OpenStreetMap")
    risk_colors = {
        "Low": "green",
        "Medium": "orange",
        "High": "red"}
    for risk_level, color in risk_colors.items():
        risk_data = map_data[
            map_data["rf_risk_level"] == risk_level
        ]
        for _, row in risk_data.iterrows():
            folium.CircleMarker(
                location=[
                    row["latitude"],
                    row["longitude"]
                ],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=(
                    f"Risk: {risk_level}<br>"
                    f"Probability: "
                    f"{row['rf_risk_probability']:.2%}<br>"
                    f"Latitude: {row['latitude']:.5f}<br>"
                    f"Longitude: {row['longitude']:.5f}"
                )
            ).add_to(arctic_map)
    st_folium(
        arctic_map,
        width=None,
        height=650)
    st.caption(
        "🟢 Low Risk | 🟠 Medium Risk | 🔴 High Risk")
    print("Interactive Arctic risk map loaded successfully.")
except Exception as e:
    st.error(f"Map generation error: {e}")
  # ============================================================
# 9. ALTERNATIVE ROUTE COMPARISON
# ============================================================
st.markdown("### Alternative Route Options")
try:
    route_summary = deployment_data.groupby("rf_risk_level").agg(route_points=("rf_risk_level","size"),average_risk=("rf_risk_probability","mean")).reset_index()
    low_risk = route_summary[route_summary["rf_risk_level"] == "Low"]
    medium_risk = route_summary[route_summary["rf_risk_level"] == "Medium"]
    high_risk = route_summary[route_summary["rf_risk_level"] == "High"]
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🟢 Lower-Risk Route")
        if not low_risk.empty:
            st.metric("Average Risk",f"{low_risk.iloc[0]['average_risk']:.2%}")
            st.write(f"Route points: {int(low_risk.iloc[0]['route_points'])}")
            st.success("Recommended based on lower predicted risk.")
        else:
            st.warning("No Low-risk route points available.")
    with col2:
        st.markdown("#### 🔴 Higher-Risk Route")
        if not high_risk.empty:
            st.metric("Average Risk",f"{high_risk.iloc[0]['average_risk']:.2%}")
            st.write(f"Route points: {int(high_risk.iloc[0]['route_points'])}")
            st.warning("Higher predicted risk. Proceed with caution.")
        else:
            st.info("No High-risk route points available.")
except Exception as e:
    st.error(f"Route comparison error: {e}")
  # ============================================================
# 10. USER ROUTE SELECTION
# ============================================================
st.markdown("### Choose Your Route")
route_options = ["Lower-Risk Route","Higher-Risk Route"]
selected_route = st.radio("Select a route:",route_options,index=0)
if selected_route == "Lower-Risk Route":
    st.success("🟢 Lower-risk route selected.")
    st.info("AI recommendation: This route has comparatively lower predicted risk.")
else:
    st.warning("🔴 Higher-risk route selected.")
    st.warning("This route has comparatively higher predicted risk. Proceed with caution.")
st.write(f"Your selected route: **{selected_route}**")
# ============================================================
# 11. FINAL ROUTE DECISION
# ============================================================
st.markdown("### Final Route Decision")
if selected_route == "Lower-Risk Route":
    final_level = "Low"
    final_message = "🟢 Recommended lower-risk route selected."
else:
    final_level = "High"
    final_message = "🔴 Higher-risk route selected. Proceed with caution."
final_data = deployment_data[deployment_data["rf_risk_level"] == final_level]
if not final_data.empty:
    final_probability = final_data["rf_risk_probability"].mean()
else:
    final_probability = float(deployment_data["rf_risk_probability"].mean())
st.success(final_message)
st.metric("Selected Route Risk",final_level)
st.metric("Average Risk Probability",f"{final_probability:.2%}")
st.info("Final route selection remains with the user.")
# ============================================================
# 12. END-TO-END DEPLOYMENT CHECK
# ============================================================
st.markdown("### Deployment Status")
checks = {
    "Random Forest model loaded": model is not None,
    "Deployment data loaded": deployment_data is not None and not deployment_data.empty,
    "Required latitude column": "latitude" in deployment_data.columns,
    "Required longitude column": "longitude" in deployment_data.columns,
    "Risk prediction available": "rf_risk_prediction" in deployment_data.columns,
    "Risk probability available": "rf_risk_probability" in deployment_data.columns,
    "Risk level available": "rf_risk_level" in deployment_data.columns,
    "Route selection available": selected_route in route_options}
for check_name, status in checks.items():
    if status:
        st.success(f"✅ {check_name}")
    else:
        st.error(f"❌ {check_name}")
deployment_ready = all(checks.values())
if deployment_ready:
    st.success("🚀 Arctic SafeRoute AI deployment pipeline is ready.")
else:
    st.error("Deployment pipeline requires further validation.")
# ============================================================
# 13. FINAL UI VALIDATION
# ============================================================
st.markdown("### Final Application Summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.metric("Route Points",len(deployment_data))
with summary_col2:
    st.metric("High-Risk Points",int((deployment_data["rf_risk_level"] == "High").sum()))
with summary_col3:
    st.metric("Low-Risk Points",int((deployment_data["rf_risk_level"] == "Low").sum()))
st.markdown("### System Features")
features = [
    "🗺️ Interactive Arctic Risk Map",
    "🟢 Low / 🟠 Medium / 🔴 High Risk Classification",
    "🛣️ Alternative Route Comparison",
    "⚠️ Higher-Risk Route Warning",
    "⭐ Lower-Risk Route Recommendation",
    "👤 User-Controlled Final Route Selection"
]
for feature in features:
    st.write(feature)
if deployment_ready:
    st.success("✅ Arctic SafeRoute AI is ready for deployment testing.")
else:
    st.warning("⚠️ Complete the failed checks before deployment testing.")
# ============================================================
# 13. FINAL UI VALIDATION
# ============================================================
st.markdown("### Final Application Summary")
summary_col1, summary_col2, summary_col3 = st.columns(3)
with summary_col1:
    st.metric("Route Points",len(deployment_data))
with summary_col2:
    st.metric("High-Risk Points",int((deployment_data["rf_risk_level"] == "High").sum()))
with summary_col3:
    st.metric("Low-Risk Points",int((deployment_data["rf_risk_level"] == "Low").sum()))
st.markdown("### System Features")
features = [
    "🗺️ Interactive Arctic Risk Map",
    "🟢 Low / 🟠 Medium / 🔴 High Risk Classification",
    "🛣️ Alternative Route Comparison",
    "⚠️ Higher-Risk Route Warning",
    "⭐ Lower-Risk Route Recommendation",
    "👤 User-Controlled Final Route Selection"
]
for feature in features:
    st.write(feature)
if deployment_ready:
    st.success("✅ Arctic SafeRoute AI is ready for deployment testing.")
else:
    st.warning("⚠️ Complete the failed checks before deployment testing.")
