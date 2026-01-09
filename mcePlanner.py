import streamlit as st
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
import calendar
import requests
from datetime import datetime, timedelta

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="MCE Planning & Feasibility Tool",
    page_icon="🔧",
    layout="wide"
)

# ========================================
# HELPER FUNCTIONS
# ========================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on Earth
    Returns distance in kilometers
    """
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    
    return c * r

def safe_df(df):
    """
    Convert dataframe to Arrow-safe format for Streamlit display
    """
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    
    df = df.copy()
    
    for col in df.columns:
        # Convert datetime to string
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
        # Convert object types to string
        elif df[col].dtype == "object":
            # Use infer_objects to avoid FutureWarning
            df[col] = df[col].fillna("")
            df = df.infer_objects(copy=False)
            df[col] = df[col].astype(str)
    
    return df

def find_nearest_crane_company(turbine_lat, turbine_lon, crane_df, required_capacity=None):
    """
    Find the nearest crane company to a turbine
    Optionally filter by required crane capacity
    """
    # Handle different column name variations
    capacity_col = None
    for col_name in ['min_capacity_t', 'Min Capacity (t)', 'min_capacity']:
        if col_name in crane_df.columns:
            capacity_col = col_name
            break
    
    # Filter by capacity if specified
    if required_capacity and capacity_col:
        available_cranes = crane_df[crane_df[capacity_col] >= required_capacity].copy()
    else:
        available_cranes = crane_df.copy()
    
    if len(available_cranes) == 0:
        return None
    
    # Calculate distances
    available_cranes["distance_km"] = available_cranes.apply(
        lambda row: haversine(turbine_lat, turbine_lon, row["latitude"], row["longitude"]),
        axis=1
    )
    
    # Sort by distance
    available_cranes = available_cranes.sort_values("distance_km")
    
    return available_cranes

def assess_feasibility(avg_wind_speed, avg_precipitation):
    """
    Assess MCE feasibility based on weather conditions
    Rules:
    - Precipitation < 15mm AND Wind Speed < 8 m/s = SUITABLE
    - Otherwise = CAUTION or NOT SUITABLE
    """
    if avg_precipitation < 15 and avg_wind_speed < 8:
        return "✅ SUITABLE", "success"
    elif avg_precipitation < 20 and avg_wind_speed < 10:
        return "⚠️ CAUTION ADVISED", "warning"
    else:
        return "❌ NOT SUITABLE", "error"

def get_weather_forecast(latitude, longitude, days=10):
    """
    Get weather forecast for next 10 days using Open-Meteo API (free, no API key needed)
    Returns two DataFrames: daily forecast and hourly forecast
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,windgusts_10m_max",
            "hourly": "temperature_2m,precipitation,windspeed_10m,windgusts_10m",
            "timezone": "Europe/Copenhagen",
            "forecast_days": days
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Create daily DataFrame
            daily_df = pd.DataFrame({
                'date': pd.to_datetime(data['daily']['time']),
                'temp_max_c': data['daily']['temperature_2m_max'],
                'temp_min_c': data['daily']['temperature_2m_min'],
                'precipitation_mm': data['daily']['precipitation_sum'],
                'wind_speed_ms': data['daily']['windspeed_10m_max'],
                'wind_gust_ms': data['daily']['windgusts_10m_max']
            })
            
            # Create hourly DataFrame
            hourly_df = pd.DataFrame({
                'datetime': pd.to_datetime(data['hourly']['time']),
                'temperature_c': data['hourly']['temperature_2m'],
                'precipitation_mm': data['hourly']['precipitation'],
                'wind_speed_ms': data['hourly']['windspeed_10m'],
                'wind_gust_ms': data['hourly']['windgusts_10m']
            })
            
            # Add date column for easier filtering
            hourly_df['date'] = hourly_df['datetime'].dt.date
            hourly_df['hour'] = hourly_df['datetime'].dt.hour
            
            return daily_df, hourly_df
        else:
            return None, None
    
    except Exception as e:
        st.warning(f"Could not fetch weather forecast: {e}")
        return None, None

def get_google_maps_url(latitude, longitude, zoom=17, map_type="satellite"):
    """
    Generate Google Maps URL for satellite view
    No API key needed for basic embed
    """
    return f"https://www.google.com/maps/@{latitude},{longitude},{zoom}z/data=!3m1!1e3"

def get_static_map_url(latitude, longitude, zoom=16):
    """
    Generate a static map image URL using OpenStreetMap tiles
    This doesn't require API key and provides a visual reference
    """
    # Using Esri World Imagery (free satellite tiles)
    # Tile math for converting lat/lon to tile coordinates
    import math
    
    def deg2num(lat_deg, lon_deg, zoom):
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)
    
    x, y = deg2num(latitude, longitude, zoom)
    
    # Esri World Imagery URL
    return f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"

def check_sentinel_availability(latitude, longitude):
    """
    Check if Sentinel-2 imagery is available for this location
    Returns info about latest available imagery
    """
    try:
        # Using Sentinel Hub's free Statistical API endpoint
        # Note: This is simplified - full implementation would need authentication
        
        # For now, we'll provide a link to Sentinel Hub EO Browser
        eo_browser_url = f"https://apps.sentinel-hub.com/eo-browser/?zoom=15&lat={latitude}&lon={longitude}&themeId=DEFAULT-THEME&visualizationUrl=https%3A%2F%2Fservices.sentinel-hub.com%2Fogc%2Fwms%2Fbd86bcc0-f318-402b-a145-015f85b9427e&datasetId=S2L2A&fromTime=2024-12-27T00%3A00%3A00.000Z&toTime=2024-12-27T23%3A59%3A59.999Z&layerId=1_TRUE_COLOR"
        
        return eo_browser_url
    except:
        return None

def calculate_precipitation_since_last_visit(last_visit_date, weather_data):
    """
    Calculate average daily precipitation since the last AP visit
    """
    try:
        # Parse the last visit date
        last_visit = pd.to_datetime(last_visit_date)
        
        # Filter weather data after last visit
        weather_data['time'] = pd.to_datetime(weather_data['time'])
        recent_weather = weather_data[weather_data['time'] > last_visit]
        
        if len(recent_weather) == 0:
            return None, None
        
        # Calculate average daily precipitation (not total)
        avg_precipitation = recent_weather['precipitation_mm'].mean()
        days_since_visit = (datetime.now() - last_visit).days
        
        return avg_precipitation, days_since_visit
    
    except Exception as e:
        return None, None

# ========================================
# DATA LOADING
# ========================================

@st.cache_data
def load_all_data():
    """
    Load all required data files with caching for performance
    """
    try:
        # Load turbine data
        turbines = pd.read_csv("TBwithinBoundary.csv")
        turbines.columns = turbines.columns.str.lower().str.strip()
        
        # Normalize column names
        column_map = {
            "lat": "latitude",
            "lon": "longitude",
            "x": "longitude",
            "y": "latitude"
        }
        turbines.rename(columns=column_map, inplace=True)
        
        # Ensure coordinates are numeric
        for col in ['latitude', 'longitude']:
            if col in turbines.columns:
                turbines[col] = pd.to_numeric(turbines[col], errors='coerce')
        
        # Load MCE cases
        mce_cases = pd.read_csv("synthetic_mce_cases.csv")
        
        # Load AP visits
        ap_visits = pd.read_csv("synthetic_ap_visits.csv")
        
        # Load crane data
        cranes = pd.read_csv("Crane_Data.csv")
        
        # AGGRESSIVE column name cleaning (for GitHub/Streamlit Cloud compatibility)
        cranes.columns = cranes.columns.str.strip()  # Remove whitespace
        cranes.columns = cranes.columns.str.lower()  # Lowercase everything
        cranes.columns = cranes.columns.str.replace(' ', '_')  # Replace spaces with underscores
        
        # Map ALL common variations to standard names
        crane_column_map = {
            "lat": "latitude",
            "lon": "longitude",
            "long": "longitude",
            "x": "longitude",
            "y": "latitude",
            "lng": "longitude",
            "company_name": "company_name",
            "min_capacity_(t)": "min_capacity_t",
            "min_capacity_t": "min_capacity_t",
            "min_capacity": "min_capacity_t",
            "daily_cost_(€)": "daily_cost_eur",
            "daily_cost_eur": "daily_cost_eur",
            "daily_cost": "daily_cost_eur",
            "max_wind_(m/s)": "max_wind_ms",
            "max_wind_ms": "max_wind_ms",
            "max_wind": "max_wind_ms",
            "mobilization_days": "mobilization_days",
            "crane_class": "crane_class"
        }
        cranes.rename(columns=crane_column_map, inplace=True)
        
        # Ensure coordinates are numeric
        for col in ['latitude', 'longitude']:
            if col in cranes.columns:
                cranes[col] = pd.to_numeric(cranes[col], errors='coerce')
        
        # Validate crane coordinates exist
        if 'latitude' not in cranes.columns or 'longitude' not in cranes.columns:
            st.error(f"""
            ❌ **Crane Data Error:** Missing coordinate columns!
            
            **Current columns (after cleaning):** {', '.join(cranes.columns.tolist())}
            
            **Original columns:** Upload your Crane_Data.csv and check the first row for exact column names.
            
            **Required:** Columns for 'latitude' and 'longitude' (or variations like lat/lon, X/Y)
            """)
            st.stop()
        
        # Load weather data (cleaned version)
        weather_holstebro = pd.read_excel("WindData_FIXED.xlsx", sheet_name="Holstebro")
        weather_ringkoebing = pd.read_excel("WindData_FIXED.xlsx", sheet_name="Ringkoebing")
        weather_skjern = pd.read_excel("WindData_FIXED.xlsx", sheet_name="Skjern")
        weather_stations = pd.read_excel("WindData_FIXED.xlsx", sheet_name="weather_stations")
        
        return (turbines, mce_cases, ap_visits, cranes, 
                weather_holstebro, weather_ringkoebing, weather_skjern, weather_stations)
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load data
turbines, mce_cases, ap_visits, cranes, weather_holstebro, weather_ringkoebing, weather_skjern, weather_stations = load_all_data()

# ========================================
# ASSIGN WEATHER STATIONS TO TURBINES
# ========================================

def assign_weather_station(row):
    """
    Assign the nearest weather station to each turbine
    """
    if pd.isna(row.get("latitude")) or pd.isna(row.get("longitude")):
        return "Holstebro"
    
    try:
        distances = weather_stations.apply(
            lambda ws: haversine(
                row["latitude"], row["longitude"],
                ws["latitude"], ws["longitude"]
            ),
            axis=1
        )
        return weather_stations.loc[distances.idxmin(), "station_id"]
    except:
        return "Holstebro"

# Assign weather stations if not already done
if "weather_station" not in turbines.columns:
    turbines["weather_station"] = turbines.apply(assign_weather_station, axis=1)

# ========================================
# USER INTERFACE
# ========================================

# Title
st.title("🔧 MCE Planning & Feasibility Tool")
st.markdown("**Major Component Exchange - Decision Support System**")
st.markdown("---")

# ========================================
# SIDEBAR - USER INPUTS
# ========================================

st.sidebar.header("📋 Planning Configuration")

# Turbine Selection
st.sidebar.subheader("1️⃣ Select Turbine")
selected_turbine = st.sidebar.selectbox(
    "Choose Turbine ID:",
    options=sorted(turbines["turbine_id"].unique()),
    index=0
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Selected Turbine:** {selected_turbine}")

# ========================================
# MAIN CONTENT
# ========================================

# Get selected turbine data
turbine_data = turbines[turbines["turbine_id"] == selected_turbine].iloc[0]

# Section 1: TURBINE INFORMATION
st.header("📍 Turbine Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Turbine ID", turbine_data["turbine_id"])
    if 'manufactur' in turbine_data.index:
        st.metric("Manufacturer", turbine_data.get('manufactur', 'N/A'))

with col2:
    if 'model' in turbine_data.index:
        st.metric("Model", turbine_data.get('model', 'N/A'))
    if 'blade length m' in turbine_data.index:
        blade_length = turbine_data.get('blade length m', 'N/A')
        st.metric("Blade Length", f"{blade_length} m" if pd.notna(blade_length) else "N/A")

with col3:
    if 'hub height' in turbine_data.index:
        hub_height = turbine_data.get('hub height', 'N/A')
        st.metric("Hub Height", f"{hub_height} m" if pd.notna(hub_height) else "N/A")
    st.metric("Weather Station", turbine_data.get('weather_station', 'N/A'))

with st.expander("🔍 View Full Turbine Details"):
    st.dataframe(safe_df(turbine_data.to_frame().T), width='stretch')

st.markdown("---")

st.header("🔧 Historical MCE Cases")

mce_for_turbine = mce_cases[mce_cases["turbine_id"] == selected_turbine]

if len(mce_for_turbine) == 0:
    st.info("ℹ️ No MCE cases recorded for this turbine")
else:
    st.success(f"**{len(mce_for_turbine)} MCE case(s) found**")
    
    # Display key information
    for idx, case in mce_for_turbine.iterrows():
        with st.expander(f"MCE: {case['mce_id']} - {case['component']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Component:**", case['component'])
                st.write("**Blade Removal:**", "Yes" if case['blade_removal'] else "No")
                st.write("**Kentledge Required:**", "Yes" if case['kentledge_required'] else "No")
            
            with col2:
                st.write("**Crane Class:**", case['crane_class'])
                st.write("**Crane Capacity:**", f"{case['crane_capacity_t']} tonnes")
                st.write("**Tech Count:**", case['tech_count'])
            
            with col3:
                st.write("**Start Date:**", case['start_date'])
                st.write("**End Date:**", case['end_date'])
                st.write("**Duration:**", f"{case['duration_days']} days")
                st.write("**Weather Delay:**", "Yes" if case['weather_delay'] else "No")
    
    # Show full table
    with st.expander("📊 View Full MCE History Data"):
        st.dataframe(safe_df(mce_for_turbine), width="stretch")

# ========================================
# HISTORICAL MCE INSIGHTS FOR PLANNING
# ========================================

st.markdown("---")
st.header("📊 Historical MCE Insights for Planning")
st.caption("Learn from past MCE cases to better plan your upcoming exchange")

# Let user select which component they're planning to exchange
st.subheader("🔧 Component Selection")

# Get all unique components from MCE history (for this turbine and all turbines)
all_components = mce_cases['component'].unique().tolist()

selected_component = st.selectbox(
    "What component are you planning to exchange?",
    options=["Select a component..."] + sorted(all_components),
    help="Select the component you're planning to exchange"
)

if selected_component != "Select a component...":
    
    st.markdown("---")
    
    # Strategy 1: Look for same component on THIS turbine
    same_turbine_same_component = mce_cases[
        (mce_cases['turbine_id'] == selected_turbine) &
        (mce_cases['component'] == selected_component)
    ]
    
    # Strategy 2: Look for same component on SAME turbine model
    turbine_model = turbine_data.get('model', None)
    turbine_manufacturer = turbine_data.get('manufactur', None)
    
    same_model_same_component = pd.DataFrame()
    if pd.notna(turbine_model):
        # Find other turbines with same model
        similar_turbines = turbines[turbines['model'] == turbine_model]['turbine_id'].tolist()
        
        same_model_same_component = mce_cases[
            (mce_cases['turbine_id'].isin(similar_turbines)) &
            (mce_cases['component'] == selected_component)
        ]
    
    # Strategy 3: Look for same component on SAME manufacturer
    same_manufacturer_same_component = pd.DataFrame()
    if pd.notna(turbine_manufacturer):
        similar_manufacturer_turbines = turbines[turbines['manufactur'] == turbine_manufacturer]['turbine_id'].tolist()
        
        same_manufacturer_same_component = mce_cases[
            (mce_cases['turbine_id'].isin(similar_manufacturer_turbines)) &
            (mce_cases['component'] == selected_component)
        ]
    
    # Display results in order of relevance
    
    # PRIORITY 1: Same turbine, same component
    if len(same_turbine_same_component) > 0:
        st.success(f"✅ Found {len(same_turbine_same_component)} historical case(s) for **{selected_component}** exchange on **this exact turbine**!")
        
        st.subheader("📋 Historical Data from This Turbine")
        
        # Get most recent case
        most_recent = same_turbine_same_component.sort_values('start_date', ascending=False).iloc[0]
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Tech Count",
                f"{most_recent['tech_count']} technicians",
                help="Number of technicians used in last exchange"
            )
        
        with col2:
            st.metric(
                "Crane Class",
                most_recent['crane_class'],
                help="Type of crane used"
            )
        
        with col3:
            st.metric(
                "Crane Capacity",
                f"{most_recent['crane_capacity_t']} tonnes",
                help="Crane capacity required"
            )
        
        with col4:
            st.metric(
                "Duration",
                f"{most_recent['duration_days']} days",
                help="Total days taken for the exchange"
            )
        
        # Additional details
        st.write("**Additional Details from Last Exchange:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Blade Removal Required:** {'Yes' if most_recent['blade_removal'] else 'No'}")
        
        with col2:
            st.write(f"**Kentledge Required:** {'Yes' if most_recent['kentledge_required'] else 'No'}")
        
        with col3:
            st.write(f"**Weather Delays:** {'Yes' if most_recent['weather_delay'] else 'No'}")
        
        st.write(f"**Last Exchange Date:** {most_recent['start_date']} to {most_recent['end_date']}")
        
        # Show all historical cases for this turbine-component combination
        if len(same_turbine_same_component) > 1:
            with st.expander(f"📜 View All {len(same_turbine_same_component)} Historical Cases"):
                st.dataframe(safe_df(same_turbine_same_component), width="stretch")
    
    # PRIORITY 2: Same model, same component
    elif len(same_model_same_component) > 0:
        st.info(f"ℹ️ No direct history on this turbine, but found {len(same_model_same_component)} case(s) on **similar turbine model** ({turbine_model})")
        
        st.subheader(f"📋 Historical Data from Similar Turbines ({turbine_model})")
        
        # Calculate averages from similar turbines
        avg_tech_count = same_model_same_component['tech_count'].mean()
        avg_duration = same_model_same_component['duration_days'].mean()
        most_common_crane = same_model_same_component['crane_class'].mode()[0] if len(same_model_same_component['crane_class'].mode()) > 0 else "N/A"
        avg_crane_capacity = same_model_same_component['crane_capacity_t'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Tech Count",
                f"{avg_tech_count:.0f} technicians",
                help="Average from similar turbines"
            )
        
        with col2:
            st.metric(
                "Most Common Crane",
                most_common_crane,
                help="Most frequently used crane class"
            )
        
        with col3:
            st.metric(
                "Avg Crane Capacity",
                f"{avg_crane_capacity:.0f} tonnes",
                help="Average crane capacity used"
            )
        
        with col4:
            st.metric(
                "Avg Duration",
                f"{avg_duration:.1f} days",
                help="Average duration for this component"
            )
        
        # Show statistics
        st.write("**Statistics from Similar Turbines:**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            blade_removal_pct = (same_model_same_component['blade_removal'].sum() / len(same_model_same_component)) * 100
            st.write(f"**Blade Removal:** {blade_removal_pct:.0f}% of cases")
        
        with col2:
            kentledge_pct = (same_model_same_component['kentledge_required'].sum() / len(same_model_same_component)) * 100
            st.write(f"**Kentledge Required:** {kentledge_pct:.0f}% of cases")
        
        with col3:
            weather_delay_pct = (same_model_same_component['weather_delay'].sum() / len(same_model_same_component)) * 100
            st.write(f"**Weather Delays:** {weather_delay_pct:.0f}% of cases")
        
        with st.expander(f"📜 View All {len(same_model_same_component)} Cases from Similar Turbines"):
            st.dataframe(safe_df(same_model_same_component), width="stretch")
    
    # PRIORITY 3: Same manufacturer, same component
    elif len(same_manufacturer_same_component) > 0:
        st.info(f"ℹ️ Found {len(same_manufacturer_same_component)} case(s) on turbines from **{turbine_manufacturer}** manufacturer")
        
        st.subheader(f"📋 Historical Data from {turbine_manufacturer} Turbines")
        
        # Calculate averages
        avg_tech_count = same_manufacturer_same_component['tech_count'].mean()
        avg_duration = same_manufacturer_same_component['duration_days'].mean()
        most_common_crane = same_manufacturer_same_component['crane_class'].mode()[0] if len(same_manufacturer_same_component['crane_class'].mode()) > 0 else "N/A"
        avg_crane_capacity = same_manufacturer_same_component['crane_capacity_t'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Avg Tech Count",
                f"{avg_tech_count:.0f} technicians"
            )
        
        with col2:
            st.metric(
                "Most Common Crane",
                most_common_crane
            )
        
        with col3:
            st.metric(
                "Avg Crane Capacity",
                f"{avg_crane_capacity:.0f} tonnes"
            )
        
        with col4:
            st.metric(
                "Avg Duration",
                f"{avg_duration:.1f} days"
            )
        
        st.caption("⚠️ Note: Data from different turbine models within same manufacturer. Use as general reference.")
        
        with st.expander(f"📜 View All {len(same_manufacturer_same_component)} Cases"):
            st.dataframe(safe_df(same_manufacturer_same_component), width="stretch")
    
    # NO DATA FOUND
    else:
        st.warning(f"⚠️ No historical data found for **{selected_component}** exchange")
        st.info("**Recommendations:**")
        st.write("- Contact manufacturer for specifications")
        st.write("- Consult with crane companies for capacity estimates")
        st.write("- Plan for longer duration as first-time exchange")
        st.write("- Ensure AP visit for ground assessment")

else:
    st.info("👆 Please select a component to see historical planning insights")

st.markdown("---")


st.header("👷 Authorized Person (AP) Visit History")

ap_for_turbine = ap_visits[ap_visits["turbine_id"] == selected_turbine]

if len(ap_for_turbine) == 0:
    st.info("ℹ️ No AP visits recorded for this turbine")
else:
    st.success(f"**{len(ap_for_turbine)} AP visit(s) found**")
    
    # Get the most recent visit
    ap_for_turbine_sorted = ap_for_turbine.sort_values('visit_date', ascending=False)
    most_recent_visit = ap_for_turbine_sorted.iloc[0]
    
    # === NEW: AP Visit Decision Support ===
    st.subheader("🎯 AP Visit Decision Support")
    
    # Get weather data for precipitation calculation
    station = turbine_data["weather_station"]
    if station == "Holstebro":
        weather_data = weather_holstebro.copy()
    elif station == "Ringkoebing":
        weather_data = weather_ringkoebing.copy()
    else:
        weather_data = weather_skjern.copy()
    
    # Calculate precipitation since last visit
    total_precip, days_since = calculate_precipitation_since_last_visit(
        most_recent_visit['visit_date'],
        weather_data
    )
    
    # Display in a highlighted box
    st.info("**📋 Last AP Visit Information**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Last Visit Date",
            most_recent_visit['visit_date'],
            help="Date of the most recent AP visit"
        )
    
    with col2:
        if days_since is not None:
            st.metric(
                "Days Since Visit",
                f"{days_since} days",
                help="Number of days since last AP visit"
            )
        else:
            st.metric("Days Since Visit", "N/A")
    
    with col3:
        if total_precip is not None:
            st.metric(
                "Avg Daily Precipitation Since Visit",
                f"{total_precip:.2f} mm/day",
                help="Average daily rainfall since the last AP visit"
            )
        else:
            st.metric("Avg Daily Precipitation Since Visit", "N/A")
    
    with col4:
        st.metric(
            "Ground Condition (Last Visit)",
            most_recent_visit['ground_condition'],
            help="Ground condition noted during last visit"
        )
    
    # Additional insights - FACTS ONLY, NO RECOMMENDATIONS
    col1, col2 = st.columns(2)
    
    with col1:
        kentledge_status = "Yes" if most_recent_visit['kentledge_recommended'] else "No"
        st.write(f"**Kentledge (Last Visit):** {kentledge_status}")
    
    with col2:
        st.write(f"**Max Crane Capacity (Last Assessment):** {most_recent_visit['max_crane_capacity_t']} tonnes")
    
    # Show notes if available
    if pd.notna(most_recent_visit['notes']) and most_recent_visit['notes'] != '':
        st.write("**Notes from Last Visit:**")
        st.info(most_recent_visit['notes'])
    
    st.markdown("---")
    
    # Display all visits in expandable sections
    st.subheader("📜 Complete AP Visit History")
    
    for idx, visit in ap_for_turbine_sorted.iterrows():
        with st.expander(f"AP Visit: {visit['ap_visit_id']} - {visit['visit_date']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**Ground Condition:**", visit['ground_condition'])
                st.write("**Kentledge Recommended:**", "Yes" if visit['kentledge_recommended'] else "No")
            
            with col2:
                st.write("**Max Crane Capacity:**", f"{visit['max_crane_capacity_t']} tonnes")
                st.write("**Exclusion Zone:**", f"{visit['exclusion_zone_m']} meters")
            
            with col3:
                st.write("**Notes:**", visit['notes'])
    
    # Show full table
    with st.expander("📊 View Full AP Visit Data"):
        st.dataframe(safe_df(ap_for_turbine_sorted), width="stretch")

st.markdown("---")

# Section 4: CRANE COMPANY RECOMMENDATION
st.header("🏗️ Crane Company Recommendation")

# Get required crane capacity from most recent MCE or AP visit
required_capacity = None
if len(mce_for_turbine) > 0:
    required_capacity = mce_for_turbine.iloc[-1]['crane_capacity_t']
elif len(ap_for_turbine) > 0:
    required_capacity = ap_for_turbine.iloc[-1]['max_crane_capacity_t']

if pd.notna(turbine_data['latitude']) and pd.notna(turbine_data['longitude']):
    nearest_cranes = find_nearest_crane_company(
        turbine_data['latitude'], 
        turbine_data['longitude'], 
        cranes,
        required_capacity=required_capacity
    )
    
    if nearest_cranes is not None and len(nearest_cranes) > 0:
        st.success(f"✅ Found {len(nearest_cranes)} suitable crane company(ies)")
        
        # Show top 3 recommendations
        top_cranes = nearest_cranes.head(3)
        
        for idx, crane in top_cranes.iterrows():
            # Get company name (handle different column names)
            company_name = crane.get('company_name', crane.get('Company Name', 'Unknown'))
            
            with st.expander(f"🏆 #{top_cranes.index.get_loc(idx) + 1}: {company_name} - {crane['distance_km']:.2f} km away"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    crane_class = crane.get('crane_class', crane.get('Crane Class', 'N/A'))
                    min_capacity = crane.get('min_capacity_t', crane.get('Min Capacity (t)', 'N/A'))
                    st.write("**Crane Class:**", crane_class)
                    st.write("**Min Capacity:**", f"{min_capacity} tonnes")
                
                with col2:
                    mobilization = crane.get('mobilization_days', crane.get('Mobilization Days', 'N/A'))
                    daily_cost = crane.get('daily_cost_eur', crane.get('Daily Cost (€)', 'N/A'))
                    st.write("**Mobilization:**", f"{mobilization} days")
                    st.write("**Daily Cost:**", f"€{daily_cost:,}" if isinstance(daily_cost, (int, float)) else daily_cost)
                
                with col3:
                    max_wind = crane.get('max_wind_ms', crane.get('Max Wind (m/s)', 'N/A'))
                    st.write("**Max Wind:**", f"{max_wind} m/s")
                    st.write("**Distance:**", f"{crane['distance_km']:.2f} km")
                
                notes = crane.get('notes', crane.get('Notes', ''))
                if pd.notna(notes) and notes != '':
                    st.write("**Notes:**", notes)
        
        # Show full table
        with st.expander("📊 View All Available Crane Companies"):
            st.dataframe(safe_df(nearest_cranes), width="stretch")
    else:
        st.warning("⚠️ No suitable crane companies found for the required capacity")
else:
    st.warning("⚠️ Cannot recommend crane company - turbine coordinates missing")

st.markdown("---")

# Section 5: WEATHER ANALYSIS & FEASIBILITY
# 10-DAY WEATHER FORECAST WITH HOURLY BREAKDOWN
# ========================================
# Section 3: MCE CASES HISTORY
st.header("🌍 10-Day Weather Forecast")
st.markdown(f"**Real-time forecast starting from {datetime.now().strftime('%B %d, %Y')}**")

# Get weather station for this turbine
forecast_station = turbine_data.get('weather_station', 'Holstebro')

# Find the weather station coordinates
station_info = weather_stations[weather_stations['station_id'] == forecast_station]

if len(station_info) > 0:
    station_lat = station_info.iloc[0]['latitude']
    station_lon = station_info.iloc[0]['longitude']
    
    st.info(f"📍 Using weather data from **{forecast_station}** station")
    st.caption(f"Coordinates: {station_lat:.4f}°N, {station_lon:.4f}°E")
    
    # Fetch forecast (now returns both daily and hourly)
    try:
        daily_forecast, hourly_forecast = get_weather_forecast(station_lat, station_lon)
        
        if daily_forecast is not None and len(daily_forecast) > 0:
            st.success("✅ Successfully retrieved 10-day forecast from Open-Meteo API!")
            
            # Display forecast metrics
            st.subheader("📊 Forecast Summary (10 Days)")
            
            # ========================================
            # NORMALIZED METRICS (Match with Historical)
            # ========================================
            
            if hourly_forecast is not None:
                # 1. Avg wind speed (all hours)
                avg_forecast_wind = hourly_forecast['wind_speed_ms'].mean()
                
                # 2. Max wind speed (highest forecasted)
                max_forecast_wind = hourly_forecast['wind_speed_ms'].max()
                
                # 3. % hours > crane limit (8 m/s)
                hours_above_limit = len(hourly_forecast[hourly_forecast['wind_speed_ms'] > 8])
                pct_above_limit = (hours_above_limit / len(hourly_forecast)) * 100
                
                # 4. Avg daily precipitation
                avg_daily_precip = daily_forecast['precipitation_mm'].mean()
                
                # 5. Consecutive feasible hours
                hourly_sorted = hourly_forecast.sort_values('datetime')
                hourly_sorted['feasible'] = (hourly_sorted['wind_speed_ms'] < 8) & (hourly_sorted['precipitation_mm'] < 0.5)
                
                current_streak = 0
                max_consecutive = 0
                for feasible in hourly_sorted['feasible']:
                    if feasible:
                        current_streak += 1
                        max_consecutive = max(max_consecutive, current_streak)
                    else:
                        current_streak = 0
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric(
                        "Avg Wind Speed",
                        f"{avg_forecast_wind:.1f} m/s",
                        help="Average across all hours in forecast"
                    )
                
                with col2:
                    st.metric(
                        "Max Wind Speed",
                        f"{max_forecast_wind:.1f} m/s",
                        help="Highest forecasted wind speed"
                    )
                
                with col3:
                    st.metric(
                        "% Hours > 8 m/s",
                        f"{pct_above_limit:.0f}%",
                        help="Percentage of hours exceeding crane limit"
                    )
                
                with col4:
                    st.metric(
                        "Avg Daily Precip",
                        f"{avg_daily_precip:.1f} mm",
                        help="Average precipitation per day"
                    )
                
                with col5:
                    st.metric(
                        "Max Consecutive Feasible",
                        f"{max_consecutive} hrs",
                        help="Longest continuous work window"
                    )
                
                # ========================================
                # COMPARISON GUIDANCE
                # ========================================
                
                st.info("""
                💡 **How to Compare with Historical Data:**
                - Scroll down to "Historical Day Analysis" section
                - Select the same day/month you're planning for
                - Compare these 5 metrics side-by-side to see if forecast is typical or unusual
                """)
            
            else:
                # Fallback to daily data only
                avg_forecast_wind = daily_forecast['wind_speed_ms'].mean()
                st.metric("Avg Wind Speed", f"{avg_forecast_wind:.1f} m/s")
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📅 Daily Overview", 
                "⏰ Hourly Forecast", 
                "🎯 Work Windows",
                "📈 Trends"
            ])
            
            # TAB 1: Daily Overview
            with tab1:
                st.subheader("Daily Forecast Details")
                
                # Format the forecast for display
                forecast_display = daily_forecast.copy()
                forecast_display['date'] = forecast_display['date'].dt.strftime('%A, %b %d')
                forecast_display.columns = [
                    'Date',
                    'Max Temp (°C)',
                    'Min Temp (°C)',
                    'Precipitation (mm)',
                    'Max Wind Speed (m/s)',
                    'Max Wind Gust (m/s)'
                ]
                
                st.dataframe(
                    safe_df(forecast_display),
                    width="stretch",
                    height=400
                )
            
            # TAB 2: Hourly Forecast
            with tab2:
                st.subheader("Hourly Weather Breakdown")
                st.caption("Select a day to see hour-by-hour conditions")
                
                if hourly_forecast is not None:
                    # Let user select a day
                    available_dates = daily_forecast['date'].dt.date.tolist()
                    selected_date = st.selectbox(
                        "Choose a day:",
                        options=available_dates,
                        format_func=lambda x: x.strftime('%A, %B %d, %Y')
                    )
                    
                    # Filter hourly data for selected day
                    day_hourly = hourly_forecast[hourly_forecast['date'] == selected_date].copy()
                    
                    if len(day_hourly) > 0:
                        # Show hourly chart
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Wind Speed Throughout the Day**")
                            st.line_chart(
                                day_hourly.set_index('hour')['wind_speed_ms'],
                                height=250
                            )
                        
                        with col2:
                            st.write("**Precipitation Throughout the Day**")
                            st.bar_chart(
                                day_hourly.set_index('hour')['precipitation_mm'],
                                height=250
                            )
                        
                        # Show hourly table
                        st.write("**Detailed Hourly Data**")
                        hourly_display = day_hourly[['hour', 'temperature_c', 'wind_speed_ms', 'wind_gust_ms', 'precipitation_mm']].copy()
                        hourly_display.columns = ['Hour', 'Temp (°C)', 'Wind (m/s)', 'Gusts (m/s)', 'Rain (mm)']
                        hourly_display['Hour'] = hourly_display['Hour'].apply(lambda x: f"{x:02d}:00")
                        
                        st.dataframe(
                            safe_df(hourly_display),
                            width="stretch",
                            height=400
                        )
                    else:
                        st.warning("No hourly data available for this day")
                else:
                    st.warning("Hourly forecast data not available")
            
            # TAB 3: Work Windows
            with tab3:
                st.subheader("🎯 Optimal Work Windows")
                st.caption("Hours with favorable conditions for MCE operations")
                
                if hourly_forecast is not None:
                    # Define favorable conditions
                    favorable_hours = hourly_forecast[
                        (hourly_forecast['wind_speed_ms'] < 8) &
                        (hourly_forecast['precipitation_mm'] < 0.5)
                    ].copy()
                    
                    if len(favorable_hours) > 0:
                        st.success(f"✅ Found {len(favorable_hours)} favorable hours in the next 10 days")
                        
                        # Group by date
                        favorable_by_date = favorable_hours.groupby('date').agg({
                            'hour': lambda x: list(x),
                            'wind_speed_ms': 'mean',
                            'precipitation_mm': 'sum'
                        }).reset_index()
                        
                        st.write("**Days with Favorable Hours:**")
                        
                        for idx, row in favorable_by_date.iterrows():
                            with st.expander(f"📅 {pd.Timestamp(row['date']).strftime('%A, %B %d')} - {len(row['hour'])} favorable hours"):
                                st.write(f"**Favorable Hours:** {', '.join([f'{h:02d}:00' for h in sorted(row['hour'])])}")
                                st.write(f"**Avg Wind Speed:** {row['wind_speed_ms']:.1f} m/s")
                                st.write(f"**Total Precipitation:** {row['precipitation_mm']:.1f} mm")
                                
                                # Show which hours are best
                                day_hours = favorable_hours[favorable_hours['date'] == row['date']]
                                best_hours = day_hours.nsmallest(5, 'wind_speed_ms')
                                
                                st.write("**🏆 Best 5 Hours:**")
                                for _, hour_row in best_hours.iterrows():
                                    st.write(f"- {hour_row['hour']:02d}:00 - Wind: {hour_row['wind_speed_ms']:.1f} m/s, Rain: {hour_row['precipitation_mm']:.1f} mm")
                    else:
                        st.warning("⚠️ No completely favorable hours found. Consider adjusting operation criteria.")
                else:
                    st.warning("Hourly forecast data not available")
            
            # TAB 4: Trends
            with tab4:
                st.subheader("📈 10-Day Forecast Trends")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Wind Speed & Gusts**")
                    st.line_chart(
                        daily_forecast.set_index('date')[['wind_speed_ms', 'wind_gust_ms']],
                        height=300
                    )
                    st.caption("Blue: Max Wind Speed | Orange: Wind Gusts")
                
                with col2:
                    st.write("**Daily Precipitation**")
                    st.bar_chart(
                        daily_forecast.set_index('date')['precipitation_mm'],
                        height=300
                    )
                    st.caption("Daily precipitation forecast")
            
            # ========================================
            # OPERATIONAL DAYS ASSESSMENT - NEW SECTION
            # ========================================
            
            st.markdown("---")
            st.header("⚙️ Operational Days Assessment")
            st.caption("Assessment based on working hours: 7 AM - 7 PM, Monday-Friday only")
            
            if hourly_forecast is not None:
                # Filter for working hours (7 AM - 7 PM) and weekdays
                hourly_forecast['datetime_full'] = pd.to_datetime(hourly_forecast['date']) + pd.to_timedelta(hourly_forecast['hour'], unit='h')
                hourly_forecast['day_of_week'] = hourly_forecast['datetime_full'].dt.dayofweek  # 0=Monday, 6=Sunday
                
                # Working hours: 7 AM (7) to 7 PM (19), Monday (0) to Friday (4)
                working_hours = hourly_forecast[
                    (hourly_forecast['hour'] >= 7) & 
                    (hourly_forecast['hour'] < 19) &
                    (hourly_forecast['day_of_week'] < 5)  # Monday to Friday
                ].copy()
                
                # Define suitable conditions
                suitable_working_hours = working_hours[
                    (working_hours['wind_speed_ms'] < 8) &
                    (working_hours['precipitation_mm'] < 0.5)
                ].copy()
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_working_hours = len(working_hours)
                    st.metric(
                        "Total Working Hours Available",
                        f"{total_working_hours} hours",
                        help="7 AM - 7 PM, Monday-Friday over next 10 days"
                    )
                
                with col2:
                    suitable_hours = len(suitable_working_hours)
                    st.metric(
                        "Suitable Working Hours",
                        f"{suitable_hours} hours",
                        help="Hours with wind < 8 m/s and rain < 0.5mm during working hours"
                    )
                
                with col3:
                    if total_working_hours > 0:
                        suitability_pct = (suitable_hours / total_working_hours) * 100
                        st.metric(
                            "Suitability %",
                            f"{suitability_pct:.1f}%"
                        )
                    else:
                        st.metric("Suitability %", "N/A")
                
                # Detailed breakdown by day
                st.markdown("---")
                st.subheader("📅 Day-by-Day Breakdown")
                
                # Group suitable hours by date
                if len(suitable_working_hours) > 0:
                    daily_breakdown = suitable_working_hours.groupby('date').agg({
                        'hour': lambda x: list(x),
                        'wind_speed_ms': 'mean',
                        'precipitation_mm': 'sum',
                        'day_of_week': 'first'
                    }).reset_index()
                    
                    # Day name mapping
                    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
                    
                    for idx, row in daily_breakdown.iterrows():
                        day_name = day_names.get(row['day_of_week'], 'Weekend')
                        date_str = pd.Timestamp(row['date']).strftime('%B %d, %Y')
                        num_hours = len(row['hour'])
                        
                        # Color code based on hours available
                        if num_hours >= 8:
                            status_icon = "✅"
                            status_color = "success"
                        elif num_hours >= 4:
                            status_icon = "⚠️"
                            status_color = "warning"
                        else:
                            status_icon = "⏰"
                            status_color = "info"
                        
                        with st.expander(f"{status_icon} {day_name}, {date_str} - {num_hours} suitable hours"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write(f"**Suitable Hours:** {', '.join([f'{h:02d}:00' for h in sorted(row['hour'])])}")
                                st.write(f"**Avg Wind Speed:** {row['wind_speed_ms']:.1f} m/s")
                                st.write(f"**Total Precipitation:** {row['precipitation_mm']:.2f} mm")
                            
                            with col2:
                                # Find best consecutive hours
                                hours_list = sorted(row['hour'])
                                consecutive_blocks = []
                                current_block = [hours_list[0]]
                                
                                for i in range(1, len(hours_list)):
                                    if hours_list[i] == hours_list[i-1] + 1:
                                        current_block.append(hours_list[i])
                                    else:
                                        consecutive_blocks.append(current_block)
                                        current_block = [hours_list[i]]
                                consecutive_blocks.append(current_block)
                                
                                # Find longest block
                                longest_block = max(consecutive_blocks, key=len)
                                
                                st.write(f"**Longest Work Window:**")
                                st.write(f"{longest_block[0]:02d}:00 - {longest_block[-1]+1:02d}:00 ({len(longest_block)} hours)")
                                
                                if len(longest_block) >= 8:
                                    st.success("🎯 Full day work possible!")
                                elif len(longest_block) >= 4:
                                    st.info("📍 Half day work possible")
                                else:
                                    st.warning("⏱️ Limited window")
                    
                    # Best days summary
                    st.markdown("---")
                    st.subheader("🏆 Recommended Work Days")
                    
                    # Sort by number of suitable hours
                    daily_breakdown['num_hours'] = daily_breakdown['hour'].apply(len)
                    best_days = daily_breakdown.nlargest(5, 'num_hours')
                    
                    st.write("**Top 5 Days for MCE Operations:**")
                    
                    for idx, day in best_days.iterrows():
                        day_name = day_names.get(day['day_of_week'], 'Weekend')
                        date_str = pd.Timestamp(day['date']).strftime('%B %d')
                        num_hours = day['num_hours']
                        
                        st.write(f"{idx + 1}. **{day_name}, {date_str}** - {num_hours} suitable hours | Wind: {day['wind_speed_ms']:.1f} m/s")
                
                else:
                    st.warning("⚠️ No completely suitable working hours found in the forecast period.")
                    st.info("💡 Consider:\n- Adjusting operation criteria\n- Postponing MCE to a different period\n- Monitoring forecast updates")
            
            else:
                st.warning("Hourly forecast data not available for operational assessment")
        
        else:
            st.error("❌ No forecast data received from Open-Meteo API")
            
    except Exception as e:
        st.error(f"❌ Error fetching weather forecast: {str(e)}")
        st.info("💡 The API call failed. This could be due to:")
        st.write("- No internet connection")
        st.write("- API rate limit reached")
        st.write("- Invalid coordinates")
        import traceback
        with st.expander("🔧 Technical Details"):
            st.code(traceback.format_exc())

else:
    st.warning(f"⚠️ Cannot fetch weather forecast - weather station '{forecast_station}' not found in database")

st.markdown("---")

# Section 5: WEATHER ANALYSIS & FEASIBILITY
st.header(f"🌤️ Historical Weather Analysis")

# Get weather data for the assigned station
station = turbine_data["weather_station"]

if station == "Holstebro":
    weather_data = weather_holstebro.copy()
elif station == "Ringkoebing":
    weather_data = weather_ringkoebing.copy()
else:
    weather_data = weather_skjern.copy()

# Process weather data
weather_data["time"] = pd.to_datetime(weather_data["time"], errors='coerce')
weather_data = weather_data[weather_data["time"].notna()]

if len(weather_data) == 0:
    st.error("❌ No valid weather data available")
else:
    weather_data["year"] = weather_data["time"].dt.year
    weather_data["month"] = weather_data["time"].dt.month
    weather_data["date"] = weather_data["time"].dt.date
    
    # ========================================
    # DATE SELECTOR FOR SPECIFIC DAY ANALYSIS
    # ========================================
    
    st.subheader("📅 Historical Day Analysis (Multi-Year)")
    st.caption("View average weather behavior for any day across multiple years")
    
    col_date1, col_date2 = st.columns(2)
    
    with col_date1:
        selected_month_analysis = st.selectbox(
            "Select Month:",
            options=list(range(1, 13)),
            format_func=lambda x: calendar.month_name[x],
            index=0,
            key="analysis_month"
        )
    
    with col_date2:
        # Get max days for selected month
        max_day = calendar.monthrange(2024, selected_month_analysis)[1]
        selected_day = st.selectbox(
            "Select Day:",
            options=list(range(1, max_day + 1)),
            index=0,
            key="analysis_day"
        )
    
    # Filter weather data for this day+month across ALL years
    daily_across_years = weather_data[
        (weather_data['time'].dt.month == selected_month_analysis) &
        (weather_data['time'].dt.day == selected_day)
    ]
    
    if len(daily_across_years) > 0:
        # Get unique years
        years_available = sorted(daily_across_years['time'].dt.year.unique())
        
        st.success(f"✅ Found data for {calendar.month_name[selected_month_analysis]} {selected_day} across {len(years_available)} years")
        st.caption(f"Years: {', '.join(map(str, years_available))}")
        
        # ========================================
        # NORMALIZED METRICS (Match with Forecast)
        # ========================================
        
        # 1. Avg wind speed (all hours)
        avg_wind_speed = daily_across_years['wind_speed_ms'].mean()
        
        # 2. Max wind speed (highest recorded)
        max_wind_speed = daily_across_years['wind_speed_ms'].max()
        
        # 3. % hours > crane limit (8 m/s)
        hours_above_limit = len(daily_across_years[daily_across_years['wind_speed_ms'] > 8])
        pct_above_limit = (hours_above_limit / len(daily_across_years)) * 100
        
        # 4. Avg daily precipitation (average total per day)
        avg_daily_precip = daily_across_years.groupby(daily_across_years['time'].dt.year)['precipitation_mm'].sum().mean()
        
        # 5. Consecutive feasible hours (longest stretch with wind < 8 and rain < 0.5)
        daily_across_years['hour'] = daily_across_years['time'].dt.hour
        
        # Calculate consecutive feasible hours
        max_consecutive_feasible = 0
        for year in years_available:
            year_data = daily_across_years[daily_across_years['time'].dt.year == year].sort_values('hour')
            year_data['feasible'] = (year_data['wind_speed_ms'] < 8) & (year_data['precipitation_mm'] < 0.5)
            
            # Find longest consecutive stretch
            current_streak = 0
            max_streak = 0
            for feasible in year_data['feasible']:
                if feasible:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            
            max_consecutive_feasible = max(max_consecutive_feasible, max_streak)
        
        # Average consecutive feasible hours across years
        consecutive_per_year = []
        for year in years_available:
            year_data = daily_across_years[daily_across_years['time'].dt.year == year].sort_values('hour')
            year_data['feasible'] = (year_data['wind_speed_ms'] < 8) & (year_data['precipitation_mm'] < 0.5)
            
            current_streak = 0
            max_streak = 0
            for feasible in year_data['feasible']:
                if feasible:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            consecutive_per_year.append(max_streak)
        
        avg_consecutive_feasible = sum(consecutive_per_year) / len(consecutive_per_year)
        
        # Display normalized metrics
        st.write("**📊 Standardized Metrics (Multi-Year Average):**")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Avg Wind Speed",
                f"{avg_wind_speed:.1f} m/s",
                help=f"Average across {len(years_available)} years"
            )
        
        with col2:
            st.metric(
                "Max Wind Speed",
                f"{max_wind_speed:.1f} m/s",
                help="Highest wind speed recorded"
            )
        
        with col3:
            st.metric(
                "% Hours > 8 m/s",
                f"{pct_above_limit:.0f}%",
                help="Percentage of hours exceeding crane limit"
            )
        
        with col4:
            st.metric(
                "Avg Daily Precip",
                f"{avg_daily_precip:.1f} mm",
                help="Average total precipitation per day"
            )
        
        with col5:
            st.metric(
                "Max Consecutive Feasible",
                f"{avg_consecutive_feasible:.0f} hrs",
                help="Average longest work window per year"
            )
        
        st.info("""
        💡 **Compare with 10-Day Forecast:**
        These metrics match the forecast format above. Compare values to assess if upcoming weather is typical or unusual for this date.
        """)
        
        # Hourly pattern across all years
        st.write(f"**Average Hourly Pattern for {calendar.month_name[selected_month_analysis]} {selected_day} (All Years):**")
        
        # Group by hour and calculate averages
        daily_across_years['hour'] = daily_across_years['time'].dt.hour
        hourly_avg = daily_across_years.groupby('hour').agg({
            'wind_speed_ms': ['mean', 'min', 'max'],
            'precipitation_mm': ['mean', 'sum']
        }).reset_index()
        
        hourly_avg.columns = ['hour', 'wind_mean', 'wind_min', 'wind_max', 'precip_mean', 'precip_sum']
        
        tab1, tab2 = st.tabs(["Wind Speed Pattern", "Precipitation Pattern"])
        
        with tab1:
            st.line_chart(
                hourly_avg.set_index('hour')[['wind_mean', 'wind_min', 'wind_max']],
                height=300
            )
            st.caption("Blue: Average | Orange: Min | Green: Max wind speed by hour")
        
        with tab2:
            st.bar_chart(
                hourly_avg.set_index('hour')['precip_mean'],
                height=300
            )
            st.caption("Average hourly precipitation (mm)")
        
        # Working hours analysis across all years
        working_hours_data = daily_across_years[
            (daily_across_years['hour'] >= 7) &
            (daily_across_years['hour'] < 19)
        ]
        
        if len(working_hours_data) > 0:
            suitable_hours_data = working_hours_data[
                (working_hours_data['wind_speed_ms'] < 8) &
                (working_hours_data['precipitation_mm'] < 0.5)
            ]
            
            # Calculate per-year suitable hours
            suitable_by_year = []
            for year in years_available:
                year_data = working_hours_data[working_hours_data['time'].dt.year == year]
                year_suitable = year_data[
                    (year_data['wind_speed_ms'] < 8) &
                    (year_data['precipitation_mm'] < 0.5)
                ]
                suitable_by_year.append(len(year_suitable))
            
            avg_suitable_hours = sum(suitable_by_year) / len(suitable_by_year)
            
            st.write(f"**Working Hours Analysis (7 AM - 7 PM):**")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.write(f"Avg suitable hours: **{avg_suitable_hours:.1f} / 12**")
                st.caption(f"Average across {len(years_available)} years")
            
            with col_b:
                working_wind_avg = working_hours_data['wind_speed_ms'].mean()
                st.write(f"Avg wind (working hrs): **{working_wind_avg:.2f} m/s**")
            
            with col_c:
                if len(working_hours_data) > 0:
                    suitability = (len(suitable_hours_data) / len(working_hours_data)) * 100
                    st.write(f"Overall suitability: **{suitability:.0f}%**")
            
            # Show year-by-year breakdown
            with st.expander("📊 Year-by-Year Breakdown"):
                year_breakdown = []
                for year in years_available:
                    year_data = daily_across_years[daily_across_years['time'].dt.year == year]
                    year_working = year_data[
                        (year_data['hour'] >= 7) &
                        (year_data['hour'] < 19)
                    ]
                    year_suitable = year_working[
                        (year_working['wind_speed_ms'] < 8) &
                        (year_working['precipitation_mm'] < 0.5)
                    ]
                    
                    year_breakdown.append({
                        'Year': year,
                        'Avg Wind (m/s)': year_data['wind_speed_ms'].mean(),
                        'Total Precip (mm)': year_data['precipitation_mm'].sum(),
                        'Suitable Hours': len(year_suitable)
                    })
                
                breakdown_df = pd.DataFrame(year_breakdown)
                st.dataframe(safe_df(breakdown_df), width="stretch")
    
    else:
        st.warning(f"⚠️ No data available for {calendar.month_name[selected_month_analysis]} {selected_day}")
    
    st.markdown("---")
    
    # ========================================
    # HISTORICAL TREND FOR SELECTED MONTH
    # ========================================
    
    st.subheader("📈 Historical Monthly Trend Analysis")
    st.caption("View historical weather trends for any month across multiple years")
    
    # Month selector for trend analysis
    selected_trend_month = st.selectbox(
        "Select Month for Trend Analysis:",
        options=list(range(1, 13)),
        format_func=lambda x: calendar.month_name[x],
        index=0,
        key="trend_month"
    )
    
    # Filter for selected month
    monthly_weather = weather_data[weather_data["month"] == selected_trend_month]
    
    if len(monthly_weather) == 0:
        st.warning(f"⚠️ No weather data available for {calendar.month_name[selected_trend_month]} at {station} station")
    else:
        # Calculate monthly averages per year (HOURLY -> DAILY -> MONTHLY AVERAGE)
        # Group by year and calculate monthly average
        monthly_avg = (
            monthly_weather
            .groupby("year")[["wind_speed_ms", "precipitation_mm"]]
            .mean()
            .reset_index()
        )
        
        # Calculate overall statistics
        avg_wind_speed = monthly_avg["wind_speed_ms"].mean()
        avg_precipitation = monthly_avg["precipitation_mm"].mean()
        years_of_data = len(monthly_avg)
        
        # Display Summary Metrics
        st.write(f"**Summary for {calendar.month_name[selected_trend_month]} (Across {years_of_data} Years):**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Avg Wind Speed",
                f"{avg_wind_speed:.2f} m/s",
                help="Average wind speed for this month across all years"
            )
        
        with col2:
            st.metric(
                "Avg Precipitation",
                f"{avg_precipitation:.2f} mm",
                help="Average precipitation for this month across all years"
            )
        
        with col3:
            st.metric(
                "Data Span",
                f"{years_of_data} years",
                help="Number of years with historical data"
            )
        
        # Historical Trend Chart
        st.markdown("---")
        st.write(f"**Year-over-Year Trend:**")
        
        # Create line chart
        st.line_chart(
            monthly_avg.set_index("year")[["wind_speed_ms", "precipitation_mm"]],
            height=400
        )
        st.caption("Blue: Wind Speed (m/s) | Orange: Precipitation (mm)")
        
        # Show detailed year-by-year data
        with st.expander("📊 View Year-by-Year Data"):
            display_data = monthly_avg.copy()
            display_data.columns = ["Year", "Avg Wind Speed (m/s)", "Avg Precipitation (mm)"]
            display_data["Avg Wind Speed (m/s)"] = display_data["Avg Wind Speed (m/s)"].round(2)
            display_data["Avg Precipitation (mm)"] = display_data["Avg Precipitation (mm)"].round(2)
            st.dataframe(safe_df(display_data), width="stretch")


# ========================================
# FOOTER
# ========================================

st.markdown("---")
st.caption("💾 Data is cached for improved performance | 🔄 Refresh page to reload data")
st.caption("🌍 Turbine locations: Holstebro, Ringkoebing, and Skjern regions, Denmark")