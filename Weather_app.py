import streamlit as st
import pandas as pd
import datetime
import random 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
 
st.set_page_config(
    page_title=" Weather Predictor", 
    layout="wide", 
)

def generate_weather_data(n=1000): 
    random.seed(42)
    
    data_dict = {
        'Temperature': [random.uniform(5, 40) for _ in range(n)],
        'Humidity': [random.uniform(30, 100) for _ in range(n)],
        'WindSpeed': [random.uniform(0, 50) for _ in range(n)],
        'Pressure': [random.gauss(1013, 10) for _ in range(n)],
    }
    data = pd.DataFrame(data_dict)
    
    rain_score = (data['Humidity'] * 0.5) - (data['Temperature'] * 0.3) - (data['Pressure'] * 0.02) + (data['WindSpeed'] * 0.1)
    
    noise = [random.gauss(0, 7) for _ in range(n)]
    
    data['RainTomorrow'] = ((rain_score + noise) > 35).astype(int) 
    return data

def train_model():
    df = generate_weather_data()
    X = df[['Temperature', 'Humidity', 'WindSpeed', 'Pressure']]
    y = df['RainTomorrow']

    if len(y.unique()) < 2:
        if 0 not in y.values:
            y.iloc[0] = 0
        if 1 not in y.values:
            y.iloc[1] = 1

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, scaler.feature_names_in_

model, scaler, feature_names = train_model()

def get_week_days():
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
    reverse_map = {v: k for k, v in DISPLAY_COLUMNS.items()}
    input_series = single_day_df.rename(columns=reverse_map).iloc[0]
    
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
        T_next = projected_data_raw['Temperature'][-1] + random.gauss(0, fluctuations['Temperature'])
        H_next = projected_data_raw['Humidity'][-1] + random.gauss(0, fluctuations['Humidity'])
        W_next = projected_data_raw['WindSpeed'][-1] + random.gauss(0, fluctuations['WindSpeed'])
        P_next = projected_data_raw['Pressure'][-1] + random.gauss(0, fluctuations['Pressure'])
        
        projected_data_raw['Temperature'].append(max(0, min(T_next, 50)))
        projected_data_raw['Humidity'].append(max(0, min(H_next, 100)))
        projected_data_raw['WindSpeed'].append(max(0, min(W_next, 70)))
        projected_data_raw['Pressure'].append(max(950, min(P_next, 1050)))

    full_df = pd.DataFrame(projected_data_raw, index=WEEK_DAYS).rename(columns=DISPLAY_COLUMNS)
    return full_df


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

def run_prediction(input_df, model, scaler):
    reverse_map = {v: k for k, v in DISPLAY_COLUMNS.items()}
    input_df_for_model = input_df.rename(columns=reverse_map)[feature_names] 
    
    scaled_input = scaler.transform(input_df_for_model)
    probabilities = model.predict_proba(scaled_input)
    predictions = model.predict(scaled_input)
    
    if probabilities.shape[1] != 2:
        rain_chances = [0] * len(input_df)
        prediction_list = ["☀️ Dry Likely"] * len(input_df)
    else:
        rain_chances = [(prob * 100).round(1) for prob in probabilities[:, 1]]
        prediction_list = ["🌧️ Rain Likely" if p == 1 else "☀️ Dry Likely" for p in predictions]
    
    results_df = pd.DataFrame({
        'Day': WEEK_DAYS,
        'Chance of Rain (%)': rain_chances,
        'Prediction': prediction_list
    })
    return results_df

results_df = run_prediction(st.session_state["forecast_data"], model, scaler)

page_options = ["🏠 Dashboard", "📊 Weekly Forecast"]
current_page_index = st.session_state["current_page_index"]
page = page_options[current_page_index]

if page == "🏠 Dashboard":
    st.title("🔮 Simple 7-Day ML Weather Predictor")
    
    st.markdown("---")

    st.subheader(f"Input Weather Data: {WEEK_DAYS[0]}")
    
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
    
    st.markdown("---")
    
    st.subheader(f"Today's Forecast: {WEEK_DAYS[0]}")
    
    first_day_data = results_df.iloc[0]
    first_day_input = st.session_state["forecast_data"].iloc[0]

    rain_chance = first_day_data['Chance of Rain (%)']
    temp = first_day_input['Temperature (°C)']
    humidity = first_day_input['Humidity (%)']
    
    icon = "☔" if rain_chance > 50 else "☀️"

    with st.container(border=True):
        col_icon, col_details = st.columns([1, 3])
        
        with col_icon:
            st.markdown(f"## {icon}") 

        with col_details:
            st.subheader(WEEK_DAYS[0])
            st.markdown(f"**Temp:** {temp:.1f}°C | **Humidity:** {humidity:.1f}% | **Rain Chance:** {rain_chance:.1f}%")
    
    st.markdown("---")
    
    if st.button("View Full 7-Day Forecast 📈", use_container_width=True):
        st.session_state["current_page_index"] = 1
        st.rerun()

elif page == "📊 Weekly Forecast":
    st.title("📊 7-Day Forecast (Line-by-Line View)") 
    
    st.markdown("---")
    
    for day_index in range(N_DAYS):
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

        high_temp = forecast['Temperature (°C)'] + random.uniform(1.0, 3.0)
        low_temp = forecast['Temperature (°C)'] - random.uniform(3.0, 5.0)
        
        with st.container(border=True):
            col_icon, col_summary = st.columns([1, 5])
            
            with col_icon:
                st.markdown(f"## {icon}")
            
            with col_summary:
                st.markdown(f"### {day}")
                
                combined_line = (
                    f"**Temperature:** {high_temp:.1f}°C (High) / {low_temp:.1f}°C (Low) "
                    f"| **Humidity:** {forecast['Humidity (%)']:.1f}% "
                    f"| **Wind:** {forecast['Wind Speed (km/h)']:.1f} km/h "
                    f"| **Pressure:** {forecast['Pressure (hPa)']:.1f} hPa "
                    f"| **Condition:** {result['Prediction']} ({rain_chance:.1f}% chance of rain)"
                )
                st.markdown(combined_line)


    
    st.markdown("---")
    if st.button("⬅️ Go Back to Dashboard", type="secondary", use_container_width=False):
        st.session_state["current_page_index"] = 0
        st.rerun()