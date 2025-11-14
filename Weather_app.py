import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
 
st.set_page_config(
    page_title="Simple 7-Day ML Weather Predictor", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🔮" 
)

def generate_weather_data(n=1000):
    np.random.seed(42)
    data = pd.DataFrame({
        'Temperature': np.random.uniform(5, 40, n),
        'Humidity': np.random.uniform(30, 100, n),
        'WindSpeed': np.random.uniform(0, 50, n),
        'Pressure': np.random.normal(1013, 10, n),
    })
    rain_score = (data['Humidity'] * 0.5) - (data['Temperature'] * 0.3) - (data['Pressure'] * 0.02) + (data['WindSpeed'] * 0.1)
    data['RainTomorrow'] = (rain_score + np.random.normal(0, 7, n) > 35).astype(int) 
    return data

@st.cache_resource
def train_model():
    df = generate_weather_data()
    X = df[['Temperature', 'Humidity', 'WindSpeed', 'Pressure']]
    y = df['RainTomorrow']

    if len(np.unique(y)) < 2:
        if 0 not in y.values:
            y.iloc[0] = 0
        if 1 not in y.values:
            y.iloc[1] = 1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    model.classes_ = np.array([0, 1]) 
    
    return model, scaler, scaler.feature_names_in_

model, scaler, feature_names = train_model()

def get_week_days():
    """Generates 7 days, starting with Today (index 0)."""
    base = datetime.date.today()
    day_list = [(base + datetime.timedelta(days=x)).strftime("%a, %b %d") for x in range(0, 7)]
    return day_list

WEEK_DAYS = get_week_days()
N_DAYS = 7

DISPLAY_COLUMNS = {
    'Temperature': 'Temperature (°C)',
    'Humidity': 'Humidity (%)',
    'WindSpeed': 'Wind Speed (km/h)',
    'Pressure': 'Pressure (hPa)',
}

def project_weather_data(single_day_df):
    """Projects the remaining 6 days based on the single input with random fluctuations."""
    reverse_map = {v: k for k, v in DISPLAY_COLUMNS.items()}
    input_series = single_day_df.rename(columns=reverse_map).iloc[0]

    np.random.seed()
    
    T_base = input_series['Temperature']
    H_base = input_series['Humidity']
    W_base = input_series['WindSpeed']
    P_base = input_series['Pressure']
    
    fluctuations = {
        'Temperature': 1.5,
        'Humidity': 5.0,
        'WindSpeed': 3.0,
        'Pressure': 1.0,
    }

    projected_data_raw = {}
    
    projected_data_raw['Temperature'] = [T_base]
    projected_data_raw['Humidity'] = [H_base]
    projected_data_raw['WindSpeed'] = [W_base]
    projected_data_raw['Pressure'] = [P_base]

    for i in range(1, N_DAYS):
        T_next = projected_data_raw['Temperature'][-1] + np.random.normal(0, fluctuations['Temperature'])
        H_next = projected_data_raw['Humidity'][-1] + np.random.normal(0, fluctuations['Humidity'])
        W_next = projected_data_raw['WindSpeed'][-1] + np.random.normal(0, fluctuations['WindSpeed'])
        P_next = projected_data_raw['Pressure'][-1] + np.random.normal(0, fluctuations['Pressure'])
        
        projected_data_raw['Temperature'].append(np.clip(T_next, 0, 50))
        projected_data_raw['Humidity'].append(np.clip(H_next, 0, 100))
        projected_data_raw['WindSpeed'].append(np.clip(W_next, 0, 70))
        projected_data_raw['Pressure'].append(np.clip(P_next, 950, 1050))

    full_df = pd.DataFrame(projected_data_raw, index=WEEK_DAYS).rename(columns=DISPLAY_COLUMNS)
    return full_df


# --- Initialization ---
initial_data_raw = {
    'Temperature': [25.0],
    'Humidity': [60.0],
    'WindSpeed': [10.0],
    'Pressure': [1013.0],
}
initial_single_day_df = pd.DataFrame(initial_data_raw, index=[WEEK_DAYS[0]]).rename(columns=DISPLAY_COLUMNS)

if "single_day_input" not in st.session_state:
    st.session_state["single_day_input"] = initial_single_day_df.copy()

if "forecast_data" not in st.session_state:
    st.session_state["forecast_data"] = project_weather_data(st.session_state["single_day_input"])

if "current_page_index" not in st.session_state:
    st.session_state["current_page_index"] = 0 
# --- End Initialization ---


def run_prediction(input_df, model, scaler):
    reverse_map = {v: k for k, v in DISPLAY_COLUMNS.items()}
    input_df_for_model = input_df.rename(columns=reverse_map)[feature_names] 
    
    scaled_input = scaler.transform(input_df_for_model)
    probabilities = model.predict_proba(scaled_input)
    
    if probabilities.shape[1] != 2:
        rain_chances = np.zeros(len(input_df))
        predictions = np.zeros(len(input_df))
    else:
        rain_chances = (probabilities[:, 1] * 100).round(1)
        predictions = model.predict(scaled_input)
    
    results_df = pd.DataFrame({
        'Day': WEEK_DAYS,
        'Chance of Rain (%)': rain_chances,
        'Prediction': np.where(predictions == 1, "🌧️ Rain Likely", "☀️ Dry Likely")
    })
    return results_df

results_df = run_prediction(st.session_state["forecast_data"], model, scaler)

# --- Sidebar Navigation ---
page_options = ["🏠 Dashboard", "📊 Weekly Forecast"]
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    page_options,
    index=st.session_state["current_page_index"]
)
st.session_state["current_page_index"] = page_options.index(page)

st.sidebar.markdown("---")
st.sidebar.caption("Model: Random Forest (scikit-learn)")

# --- Page Logic ---

if page == "🏠 Dashboard":
    st.title("🔮 Simple 7-Day ML Weather Predictor")
    st.markdown("""
        ### Predict the Current Day, Forecast the Week
        Enter the **current day's** key weather parameters below. The application will use these conditions to project the weather for the next six days and predict the likelihood of rain for the full week.
        """)
    
    st.markdown("---")

    st.subheader(f"1. Input Current Day's Weather Data ({WEEK_DAYS[0]})")
    
    # --- Input Form for the first day ---
    current_input = st.session_state["single_day_input"].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        new_temp = st.number_input(
            "🌡️ Temperature (°C)",
            min_value=0.0, max_value=45.0, step=0.1, format="%.1f",
            value=float(current_input['Temperature (°C)'])
        )

        new_wind = st.number_input(
            "💨 Wind Speed (km/h)",
            min_value=0.0, max_value=60.0, step=0.1, format="%.1f",
            value=float(current_input['Wind Speed (km/h)'])
        )

    with col2:
        new_humidity = st.number_input(
            "💧 Humidity (%)",
            min_value=0, max_value=100, step=1, format="%d",
            value=int(current_input['Humidity (%)'])
        )

        new_pressure = st.number_input(
            "📉 Pressure (hPa)",
            min_value=950.0, max_value=1050.0, step=0.1, format="%.1f",
            value=float(current_input['Pressure (hPa)'])
        )
    
    new_data_raw = {
        'Temperature (°C)': [new_temp],
        'Humidity (%)': [new_humidity],
        'Wind Speed (km/h)': [new_wind],
        'Pressure (hPa)': [new_pressure],
    }
    
    new_single_day_df = pd.DataFrame(new_data_raw, index=[WEEK_DAYS[0]])


    if not new_single_day_df.round(1).equals(st.session_state["single_day_input"].round(1)):
        st.session_state["single_day_input"] = new_single_day_df
        st.session_state["forecast_data"] = project_weather_data(new_single_day_df)
        st.toast("✅ Forecast updated!", icon='☔')
        st.rerun() 
    
    st.info("The next six days are automatically projected from this input to simulate a week's weather trend.")
    st.markdown("---")
    
    st.subheader(f"2. Current Day Forecast Summary: {WEEK_DAYS[0]}")
    
    first_day_data = results_df.iloc[0]
    first_day_input = st.session_state["forecast_data"].iloc[0]

    rain_chance = first_day_data['Chance of Rain (%)']
    temp = first_day_input['Temperature (°C)']
    humidity = first_day_input['Humidity (%)']
    
    icon = "☔" if rain_chance > 50 else "☀️"
    status_text = "High chance of rain." if rain_chance > 70 else ("Moderate risk of showers." if rain_chance > 40 else "Looks like a dry day!")

    st.container(border=True).markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px;">
            <div style="font-size: 5rem;">{icon}</div>
            <div style="flex-grow: 1; margin-left: 20px;">
                <h3 style="margin-top: 0; color: #4CAF50;">{WEEK_DAYS[0]}</h3>
                <p style="font-size: 1.25rem; font-weight: bold;">{status_text}</p>
                <p>Temp: **{temp:.1f}°C** | Humidity: **{humidity:.1f}%** | Rain Chance: **{rain_chance:.1f}%**</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("View Full 7-Day Forecast 📈", type="primary", use_container_width=True):
        st.session_state["current_page_index"] = 1
        st.rerun()

elif page == "📊 Weekly Forecast":
    st.title("📊 7-Day Forecast (Card View)")
    st.caption("Review the detailed, day-by-day forecast based on your input data.")

    st.markdown("---")
    
    cards_per_row = 3
    
    for i in range(0, N_DAYS, cards_per_row):
        cols = st.columns(cards_per_row)
        
        for j in range(cards_per_row):
            day_index = i + j
            
            if day_index < N_DAYS:
                with cols[j]:
                    day = WEEK_DAYS[day_index]
                    forecast = st.session_state["forecast_data"].iloc[day_index]
                    result = results_df.iloc[day_index]
                    
                    rain_chance = result['Chance of Rain (%)']
                    
                    if rain_chance > 70:
                        icon = "⛈️"
                    elif rain_chance > 30:
                        icon = "🌧️"
                    elif forecast['Temperature (°C)'] > 30:
                        icon = "🔥"
                    elif forecast['Temperature (°C)'] > 20:
                        icon = "☀️"
                    else:
                        icon = "☁️"

                    with st.container(border=True):
                        st.markdown(f"### {day.split(',')[0]}") 
                        st.caption(day.split(',')[1].strip()) 
                        
                        st.markdown(f"<div style='font-size: 3.5rem; text-align: center; padding: 10px 0;'>{icon}</div>", unsafe_allow_html=True)
                        
                        high_temp = forecast['Temperature (°C)'] + np.random.uniform(1.0, 3.0)
                        low_temp = forecast['Temperature (°C)'] - np.random.uniform(3.0, 5.0)
                        
                        st.markdown(f"**High:** **{high_temp:.1f}°C**")
                        st.markdown(f"<span style='color: gray;'>Low: {low_temp:.1f}°C</span>", unsafe_allow_html=True)
                        
                        st.divider()
                        
                        st.markdown(f"**Rain Chance:** <span style='color: {'#ef4444' if rain_chance > 30 else '#3b82f6'}; font-weight: bold;'>{rain_chance:.1f}%</span>", unsafe_allow_html=True)
                        st.markdown(f"**Humidity:** {forecast['Humidity (%)']:.1f}%")
                        st.markdown(f"**Wind:** {forecast['Wind Speed (km/h)']:.1f} km/h")
                        st.markdown(f"**Pressure:** {forecast['Pressure (hPa)']:.1f} hPa")

    
    st.markdown("---")
    st.info("The ML model predicts the likelihood of rain, and the card's main icon is chosen based on that probability and the projected temperature.")