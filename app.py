import streamlit as st
import pandas as pd
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Energy Consumption Analytics Dashboard",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚡ Energy Consumption Analytics Dashboard")

st.markdown(
    """
    This dashboard analyzes energy consumption patterns and uses
    **Logistic Regression** to predict whether the energy consumption
    is **High** or **Normal** based on user-entered values.
    """
)


# ============================================================
# LOAD DATASET
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")


data = load_data()


# ============================================================
# BASIC DATA PREPARATION
# ============================================================

# Remove completely empty rows if any
data = data.dropna(how="all")


# Convert DayOfWeek into numerical values if required
day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

if data["DayOfWeek"].dtype == "object":
    data["DayOfWeek"] = data["DayOfWeek"].map(day_mapping)


# ============================================================
# CREATE TARGET VARIABLE
# ============================================================

# Median energy consumption is used as the threshold.
# Above/equal to median -> High
# Below median -> Normal

energy_threshold = data["EnergyConsumption"].median()

data["ConsumptionLevel"] = (
    data["EnergyConsumption"] >= energy_threshold
).astype(int)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Manual Input")

st.sidebar.markdown(
    "Enter the energy-related values below and click "
    "**Predict Consumption**."
)


# ------------------------------------------------------------
# Month
# ------------------------------------------------------------

available_months = sorted(data["Month"].dropna().unique())

month = st.sidebar.selectbox(
    "Month",
    available_months
)


# ------------------------------------------------------------
# Hour
# ------------------------------------------------------------

hour = st.sidebar.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=12
)


# ------------------------------------------------------------
# Day of Week
# ------------------------------------------------------------

day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    day_names
)


# ------------------------------------------------------------
# Holiday
# ------------------------------------------------------------

holiday = st.sidebar.selectbox(
    "Holiday",
    ["No", "Yes"]
)


# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

temperature = st.sidebar.number_input(
    "Temperature",
    min_value=float(data["Temperature"].min()),
    max_value=float(data["Temperature"].max()),
    value=float(data["Temperature"].mean())
)


# ------------------------------------------------------------
# Humidity
# ------------------------------------------------------------

humidity = st.sidebar.number_input(
    "Humidity",
    min_value=float(data["Humidity"].min()),
    max_value=float(data["Humidity"].max()),
    value=float(data["Humidity"].mean())
)


# ------------------------------------------------------------
# Square Footage
# ------------------------------------------------------------

square_footage = st.sidebar.number_input(
    "Square Footage",
    min_value=float(data["SquareFootage"].min()),
    max_value=float(data["SquareFootage"].max()),
    value=float(data["SquareFootage"].mean())
)


# ------------------------------------------------------------
# Occupancy
# ------------------------------------------------------------

occupancy = st.sidebar.slider(
    "Occupancy",
    min_value=int(data["Occupancy"].min()),
    max_value=int(data["Occupancy"].max()),
    value=int(data["Occupancy"].mean())
)


# ------------------------------------------------------------
# HVAC Usage
# ------------------------------------------------------------

hvac_usage = st.sidebar.selectbox(
    "HVAC Usage",
    ["Off", "On"]
)


# ------------------------------------------------------------
# Lighting Usage
# ------------------------------------------------------------

lighting_usage = st.sidebar.selectbox(
    "Lighting Usage",
    ["Off", "On"]
)


# ------------------------------------------------------------
# Renewable Energy
# ------------------------------------------------------------

renewable_energy = st.sidebar.number_input(
    "Renewable Energy",
    min_value=float(data["RenewableEnergy"].min()),
    max_value=float(data["RenewableEnergy"].max()),
    value=float(data["RenewableEnergy"].mean())
)


# ============================================================
# DASHBOARD KPIs
# ============================================================

st.subheader("📊 Energy Consumption Overview")

col1, col2, col3, col4 = st.columns(4)


with col1:
    st.metric(
        "Total Records",
        len(data)
    )


with col2:
    st.metric(
        "Average Energy",
        f"{data['EnergyConsumption'].mean():.2f}"
    )


with col3:
    st.metric(
        "Maximum Energy",
        f"{data['EnergyConsumption'].max():.2f}"
    )


with col4:
    st.metric(
        "Average Temperature",
        f"{data['Temperature'].mean():.2f}"
    )


# ============================================================
# CHART 1
# AVERAGE ENERGY CONSUMPTION BY HOUR
# COLOR: BLUE
# ============================================================

st.subheader("⏰ Average Energy Consumption by Hour")

hourly_energy = (
    data.groupby("Hour")["EnergyConsumption"]
    .mean()
    .reset_index()
)

fig1 = px.line(
    hourly_energy,
    x="Hour",
    y="EnergyConsumption",
    markers=True,
    title="Average Energy Consumption by Hour"
)

fig1.update_traces(
    line=dict(
        color="#3498DB",
        width=3
    ),
    marker=dict(
        color="#1F618D",
        size=8
    )
)

fig1.update_layout(
    xaxis_title="Hour",
    yaxis_title="Average Energy Consumption",
    template="plotly_white"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# ============================================================
# CHART 2
# AVERAGE ENERGY BY DAY
# COLOR: GREEN
# ============================================================

st.subheader("📅 Average Energy Consumption by Day")

# Convert numerical day back to name for visualization

day_display_mapping = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

day_energy = (
    data.groupby("DayOfWeek")["EnergyConsumption"]
    .mean()
    .reset_index()
)

day_energy["Day"] = day_energy["DayOfWeek"].map(
    day_display_mapping
)

day_energy = day_energy.sort_values("DayOfWeek")


fig2 = px.bar(
    day_energy,
    x="Day",
    y="EnergyConsumption",
    title="Average Energy Consumption by Day"
)

fig2.update_traces(
    marker_color="#2ECC71"
)

fig2.update_layout(
    xaxis_title="Day of Week",
    yaxis_title="Average Energy Consumption",
    template="plotly_white"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# ============================================================
# CHART 3
# AVERAGE ENERGY BY OCCUPANCY
# COLOR: PURPLE
# ============================================================

st.subheader("👥 Average Energy Consumption by Occupancy")

occupancy_energy = (
    data.groupby("Occupancy")["EnergyConsumption"]
    .mean()
    .reset_index()
)

fig3 = px.bar(
    occupancy_energy,
    x="Occupancy",
    y="EnergyConsumption",
    title="Average Energy Consumption by Occupancy"
)

fig3.update_traces(
    marker_color="#9B59B6"
)

fig3.update_layout(
    xaxis_title="Occupancy",
    yaxis_title="Average Energy Consumption",
    template="plotly_white"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)


# ============================================================
# CHART 4
# TEMPERATURE VS ENERGY CONSUMPTION
# COLOR: ORANGE
# ============================================================

st.subheader("🌡️ Temperature vs Energy Consumption")

fig4 = px.scatter(
    data,
    x="Temperature",
    y="EnergyConsumption",
    title="Temperature vs Energy Consumption",
    opacity=0.7
)

fig4.update_traces(
    marker=dict(
        color="#E67E22",
        size=8
    )
)

fig4.update_layout(
    xaxis_title="Temperature",
    yaxis_title="Energy Consumption",
    template="plotly_white"
)

st.plotly_chart(
    fig4,
    use_container_width=True
)


# ============================================================
# CHART 5
# RENEWABLE ENERGY VS CONSUMPTION
# COLOR: TEAL
# ============================================================

st.subheader("🌱 Renewable Energy vs Energy Consumption")

fig5 = px.scatter(
    data,
    x="RenewableEnergy",
    y="EnergyConsumption",
    title="Renewable Energy vs Energy Consumption",
    opacity=0.7
)

fig5.update_traces(
    marker=dict(
        color="#1ABC9C",
        size=8
    )
)

fig5.update_layout(
    xaxis_title="Renewable Energy",
    yaxis_title="Energy Consumption",
    template="plotly_white"
)

st.plotly_chart(
    fig5,
    use_container_width=True
)


# ============================================================
# CHART 6
# HVAC USAGE VS ENERGY
# COLOR: RED
# ============================================================

st.subheader("❄️ HVAC Usage vs Average Energy Consumption")

hvac_energy = (
    data.groupby("HVACUsage")["EnergyConsumption"]
    .mean()
    .reset_index()
)

fig6 = px.bar(
    hvac_energy,
    x="HVACUsage",
    y="EnergyConsumption",
    title="HVAC Usage vs Average Energy Consumption",
    color="HVACUsage",
    color_discrete_map={
        "Off": "#95A5A6",
        "On": "#E74C3C"
    }
)

fig6.update_layout(
    xaxis_title="HVAC Usage",
    yaxis_title="Average Energy Consumption",
    template="plotly_white",
    showlegend=False
)

st.plotly_chart(
    fig6,
    use_container_width=True
)


# ============================================================
# STATISTICAL SUMMARY
# ============================================================

st.subheader("📋 Statistical Summary")

summary_columns = [
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "RenewableEnergy",
    "EnergyConsumption"
]

st.dataframe(
    data[summary_columns].describe(),
    use_container_width=True
)


# ============================================================
# MACHINE LEARNING
# LOGISTIC REGRESSION
# ============================================================

st.subheader("🤖 Energy Consumption Prediction")


# ------------------------------------------------------------
# SELECT FEATURES
# ------------------------------------------------------------

X = data[
    [
        "Month",
        "Hour",
        "DayOfWeek",
        "Holiday",
        "Temperature",
        "Humidity",
        "SquareFootage",
        "Occupancy",
        "HVACUsage",
        "LightingUsage",
        "RenewableEnergy"
    ]
]

y = data["ConsumptionLevel"]


# ------------------------------------------------------------
# ONE-HOT ENCODING
# ------------------------------------------------------------

X = pd.get_dummies(
    X,
    columns=[
        "DayOfWeek",
        "Holiday",
        "HVACUsage",
        "LightingUsage"
    ],
    dtype=int
)


# ------------------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ------------------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# LOGISTIC REGRESSION MODEL
# ------------------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# MODEL EVALUATION
# ============================================================

y_pred = model.predict(X_test_scaled)


accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


# ============================================================
# DISPLAY MODEL METRICS
# ============================================================

st.markdown("### 📈 Model Performance")

metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )


with metric2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )


with metric3:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )


with metric4:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown("### 🔲 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Normal", "Actual High"],
    columns=["Predicted Normal", "Predicted High"]
)

st.dataframe(
    cm_df,
    use_container_width=True
)


# ============================================================
# MANUAL PREDICTION
# ============================================================

st.sidebar.markdown("---")

predict_button = st.sidebar.button(
    "🔮 Predict Consumption",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # CREATE MANUAL INPUT DATAFRAME
    # --------------------------------------------------------

    manual_data = pd.DataFrame(
        {
            "Month": [month],
            "Hour": [hour],
            "DayOfWeek": [day_mapping[day_of_week]],
            "Holiday": [holiday],
            "Temperature": [temperature],
            "Humidity": [humidity],
            "SquareFootage": [square_footage],
            "Occupancy": [occupancy],
            "HVACUsage": [hvac_usage],
            "LightingUsage": [lighting_usage],
            "RenewableEnergy": [renewable_energy]
        }
    )


    # --------------------------------------------------------
    # APPLY SAME ENCODING
    # --------------------------------------------------------

    manual_data = pd.get_dummies(
        manual_data,
        columns=[
            "DayOfWeek",
            "Holiday",
            "HVACUsage",
            "LightingUsage"
        ],
        dtype=int
    )


    # --------------------------------------------------------
    # MAKE SURE COLUMNS MATCH TRAINING DATA
    # --------------------------------------------------------

    manual_data = manual_data.reindex(
        columns=X_train.columns,
        fill_value=0
    )


    # --------------------------------------------------------
    # SCALE MANUAL INPUT
    # --------------------------------------------------------

    manual_data_scaled = scaler.transform(
        manual_data
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        manual_data_scaled
    )[0]


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        manual_data_scaled
    )[0]


    # Find probability of class 1 = High
    high_class_index = list(
        model.classes_
    ).index(1)

    normal_class_index = list(
        model.classes_
    ).index(0)

    high_probability = probabilities[
        high_class_index
    ]

    normal_probability = probabilities[
        normal_class_index
    ]


    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    st.markdown("---")

    st.subheader("🔮 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ HIGH ENERGY CONSUMPTION"
        )

    else:

        st.success(
            "✅ NORMAL ENERGY CONSUMPTION"
        )


    # --------------------------------------------------------
    # PROBABILITY METRICS
    # --------------------------------------------------------

    result_col1, result_col2 = st.columns(2)


    with result_col1:

        st.metric(
            "Normal Probability",
            f"{normal_probability * 100:.2f}%"
        )


    with result_col2:

        st.metric(
            "High Probability",
            f"{high_probability * 100:.2f}%"
        )


    # --------------------------------------------------------
    # PROBABILITY CHART
    # --------------------------------------------------------

    probability_data = pd.DataFrame(
        {
            "Consumption Level": [
                "Normal",
                "High"
            ],
            "Probability": [
                normal_probability * 100,
                high_probability * 100
            ]
        }
    )


    probability_fig = px.bar(
        probability_data,
        x="Consumption Level",
        y="Probability",
        title="Prediction Probability",
        color="Consumption Level",
        color_discrete_map={
            "Normal": "#2ECC71",
            "High": "#E74C3C"
        }
    )


    probability_fig.update_layout(
        yaxis_title="Probability (%)",
        xaxis_title="Consumption Level",
        template="plotly_white",
        showlegend=False
    )


    st.plotly_chart(
        probability_fig,
        use_container_width=True
    )


    # --------------------------------------------------------
    # THRESHOLD
    # --------------------------------------------------------

    st.info(
        f"Energy consumption threshold: "
        f"{energy_threshold:.2f}"
    )


    # --------------------------------------------------------
    # DISPLAY USER INPUT
    # --------------------------------------------------------

    st.markdown("### 📝 Entered Values")

    entered_values = pd.DataFrame(
        {
            "Parameter": [
                "Month",
                "Hour",
                "Day",
                "Holiday",
                "Temperature",
                "Humidity",
                "Square Footage",
                "Occupancy",
                "HVAC Usage",
                "Lighting Usage",
                "Renewable Energy"
            ],
            "Value": [
                month,
                hour,
                day_of_week,
                holiday,
                temperature,
                humidity,
                square_footage,
                occupancy,
                hvac_usage,
                lighting_usage,
                renewable_energy
            ]
        }
    )


    st.dataframe(
        entered_values,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "⚡ Energy Consumption Analytics Dashboard | "
    "Python + Pandas + Streamlit + Plotly + Logistic Regression"
)