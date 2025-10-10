import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import altair as alt

API_KEY = "7b7ad2-eba700"
BASE_URL = "https://aviation-edge.com/v2/public"

# ================================
# Utility functions
# ================================
def fetch_flights(airline_code=None):
    url = f"{BASE_URL}/flights?key={API_KEY}"
    if airline_code:
        url += f"&airlineIata={airline_code}"
    response = requests.get(url)
    return response.json()
def fetch_airport_timetable(iata_code, flight_type):
    url = f"{BASE_URL}/timetable?key={API_KEY}&iataCode={iata_code}&type={flight_type}"
    response = requests.get(url)
    return response.json()

def fetch_flights_history(code=None, type_filter=None, date_from=None, date_to=None):
    url = f"{BASE_URL}/flightsHistory?key={API_KEY}"
    if code:
        url += f"&code={code}"
    if type_filter:
        url += f"&type={type_filter}"
    if date_from:
        url += f"&date_from={date_from}"
    if date_to:
        url += f"&date_to={date_to}"

    response = requests.get(url)
    try:
        return response.json()
    except Exception as e:
        return {"error": f"Failed to parse JSON: {e}"}

def fetch_airports():
    url = f"{BASE_URL}/airportDatabase?key={API_KEY}"
    return requests.get(url).json()

def fetch_airlines():
    url = f"{BASE_URL}/airlineDatabase?key={API_KEY}"
    return requests.get(url).json()

def fetch_countries():
    url = f"{BASE_URL}/countries?key={API_KEY}"
    return requests.get(url).json()

# ================================
# Dashboard Functions
# ================================
def dashboard_flight_map():
    st.title("🛫 Real-Time Flight Map")
    st.write("Live flight positions around the world.")

    flights = fetch_flights()
    data = []
    for f in flights:
        try:
            data.append({
                "latitude": float(f["geography"]["latitude"]),
                "longitude": float(f["geography"]["longitude"]),
            })
        except:
            continue
    df = pd.DataFrame(data)
    st.map(df)
    st.markdown(f"<h3 style='color: inherit;'>Total live flights: {len(df)}</h3>", unsafe_allow_html=True)


def dashboard_airport_timetable():
    st.title("📋 Airport Timetable")
    iata = st.text_input("Enter Airport IATA Code (e.g., AMM):")
    flight_type = st.selectbox("Flight Type", ["arrival", "departure"])

    if iata:
        data = fetch_airport_timetable(iata.upper(), flight_type)
        if isinstance(data, list) and data:
            rows = []
            for f in data:
                try:
                    time_info = f[flight_type].get("scheduledTime", None)
                    rows.append({
                        "Flight Number": f["flight"].get("iataNumber"),
                        "Status": f.get("status", "-"),
                        "Scheduled Time": time_info
                    })
                except:
                    continue
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No data available or invalid IATA code.")

def dashboard_flight_delay_statistics():
    st.title("⏰ Flight Delay Statistics")
    iata = st.text_input("Enter Airport IATA Code:")

    if iata:
        data = fetch_airport_timetable(iata, "departure")
        rows = []
        for f in data:
            try:
                scheduled = pd.to_datetime(f["departure"]["scheduledTime"])
                estimated = pd.to_datetime(f["departure"].get("estimatedTime", scheduled))
                delay = (estimated - scheduled).total_seconds() / 60
                rows.append({
                    "Flight": f["flight"]["iataNumber"],
                    "Delay (min)": round(delay, 2)
                })
            except:
                continue

        if rows:
            df = pd.DataFrame(rows)
            st.subheader("📊 Delay by Flight")
            st.bar_chart(df.set_index("Flight"))

            # Average delay box
            avg_delay = df["Delay (min)"].mean()
            st.markdown(
                f"""
                <div style='
                    background-color: #1e1e1e;
                    border-radius: 12px;
                    padding: 20px;
                    text-align: center;
                    margin-top: 20px;
                    margin-bottom: 20px;
                    border: 1px solid #444;'>
                    <div style='font-size: 24px; font-weight: bold; color: white;'>Average Delay (min)</div>
                    <div style='font-size: 40px; font-weight: 900; color: white;'>{avg_delay:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Donut Chart: On-Time vs Delayed
            df["On Time"] = df["Delay (min)"] <= 0
            summary = df["On Time"].value_counts().rename({True: "On Time", False: "Delayed"}).reset_index()
            summary.columns = ["Status", "Count"]

            st.markdown(
                """
                <div style='font-size: 24px; font-weight: bold; color: white; text-align: center; margin-top: 10px;'>
                    On-Time vs Delayed Flights
                </div>
                """,
                unsafe_allow_html=True
            )

            fig = px.pie(summary, names="Status", values="Count", hole=0.5)
            fig.update_layout(
                showlegend=True,
                legend_title=None,
                margin=dict(t=10, b=10, l=0, r=0),
                font=dict(color="white"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("No valid delay data found.")

def dashboard_historical_traffic_trends():
    st.title("📉 Historical Flight Traffic Trends")

    with st.form("airport_form"):
        iata = st.text_input("Enter Airport IATA Code (e.g., JFK):")
        col1, col2 = st.columns(2)
        with col1:
            date_from = st.date_input("From (UTC)", value=datetime.utcnow().date() - timedelta(days=10))
        with col2:
            date_to = st.date_input("To (UTC)", value=datetime.utcnow().date() - timedelta(days=3))
        submitted = st.form_submit_button("Submit")

    if not submitted or not iata:
        return

    if (date_to - date_from).days > 10:
        st.warning("Please select a time range of 10 days or less.")
        return

    # Convert to string
    date_from_str = date_from.strftime("%Y-%m-%d")
    date_to_str = date_to.strftime("%Y-%m-%d")

    # Fetch data
    with st.spinner("Fetching flight history data..."):
        try:
            data_departures = fetch_flights_history(code=iata, type_filter="departure", date_from=date_from_str, date_to=date_to_str)
            data_arrivals = fetch_flights_history(code=iata, type_filter="arrival", date_from=date_from_str, date_to=date_to_str)

            st.success(f"Fetched {len(data_departures)} departures and {len(data_arrivals)} arrivals.")
        except Exception as e:
            st.error(f"Failed to fetch data: {e}")
            return

    def process_data(data, key):
        rows = []
        for f in data:
            try:
                date = pd.to_datetime(f[key]["scheduledTime"])
                airline = f.get("airline", {}).get("name", "Unknown")
                rows.append({"Date": date.date(), "Hour": date.hour, "Airline": airline})
            except:
                continue
        return pd.DataFrame(rows)

    df_dep = process_data(data_departures, "departure")
    df_arr = process_data(data_arrivals, "arrival")

    if df_dep.empty and df_arr.empty:
        st.warning("No valid historical data found for this airport.")
        return

    # Daily flights chart
    def plot_daily_flights(df_dep, df_arr):
        dep_daily = df_dep.groupby("Date").size().rename("Departures")
        arr_daily = df_arr.groupby("Date").size().rename("Arrivals")
        combined = pd.concat([dep_daily, arr_daily], axis=1).fillna(0)
        st.subheader("📆 Daily Departures and Arrivals")
        st.line_chart(combined)

    # Top airlines
    def plot_top_airlines(df, label):
        airline_counts = df["Airline"].value_counts().nlargest(5).reset_index()
        airline_counts.columns = ["Airline", "Flights"]
        chart = alt.Chart(airline_counts).mark_bar().encode(
            x=alt.X('Flights:Q'),
            y=alt.Y('Airline:N', sort='-x'),
            tooltip=['Airline', 'Flights']
        ).properties(width=600, height=300)
        st.subheader(f"✈️ Top 5 Airlines ({label})")
        st.altair_chart(chart, use_container_width=True)

    # Hourly traffic
    def plot_hourly_distribution(df, label):
        hourly = df["Hour"].value_counts().sort_index().reset_index()
        hourly.columns = ["Hour", "Flights"]
        chart = alt.Chart(hourly).mark_bar().encode(
            x=alt.X('Hour:O', title='Hour (UTC)'),
            y=alt.Y('Flights:Q'),
            tooltip=['Hour', 'Flights']
        ).properties(width=600, height=300)
        st.subheader(f"🕐 Peak {label} Hours (UTC)")
        st.altair_chart(chart, use_container_width=True)

    # Render
    plot_daily_flights(df_dep, df_arr)

    if not df_dep.empty:
        plot_top_airlines(df_dep, "Departures")
        plot_hourly_distribution(df_dep, "Departure")

    if not df_arr.empty:
        plot_top_airlines(df_arr, "Arrivals")
        plot_hourly_distribution(df_arr, "Arrival")



def dashboard_airlines():
    st.title("🛫 Airline Activities Dashboard")

    airlines_data = fetch_airlines()

    if not airlines_data or not isinstance(airlines_data, list):
        st.error("No airline data returned from API.")
        return

    # Filter valid airlines
    valid_airlines = []
    airline_map = {}
    for a in airlines_data:
        name = a.get("nameAirline")
        iata = a.get("codeIataAirline")
        icao = a.get("codeIcaoAirline")
        code = iata or icao
        if name and code:
            label = f"{name} ({code})"
            valid_airlines.append(label)
            airline_map[label] = a

    if not valid_airlines:
        st.error("No valid airlines found after filtering.")
        return

    valid_airlines.sort()
    selected_label = st.selectbox("Select an Airline", valid_airlines)
    selected_airline = airline_map[selected_label]
    iata_code = selected_airline.get("codeIataAirline")

    st.markdown("## 📊 Airline Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📦 Fleet Size", selected_airline.get("sizeAirline", "N/A"))
        st.metric("📍 Country", selected_airline.get("nameCountry", "N/A"))
        st.metric("🏁 Hub Airport", selected_airline.get("codeHub", "N/A"))
    with col2:
        st.metric("🕰️ Avg Fleet Age", f"{selected_airline.get('ageFleet', 'N/A')} years")
        st.metric("📅 Founded", selected_airline.get("founding", "N/A"))
        st.metric("🔧 Type", f"{selected_airline.get('type', 'N/A')} / {selected_airline.get('statusAirline', 'N/A')}")

    # Use the existing fetch_flights with airline_code parameter
    flights = fetch_flights(airline_code=iata_code)

    if not flights or not isinstance(flights, list) or len(flights) == 0:
        st.warning("No flights found for this airline.")
        return

    # 🗺️ Destination Airports Map
    st.markdown("## 🗺️ Live Flights Tracking")
    dest_coords = []
    for f in flights:
        geo = f.get("geography", {})
        lat = geo.get("latitude")
        lon = geo.get("longitude")
        if lat and lon:
            dest_coords.append({"lat": lat, "lon": lon})
    if dest_coords:
        st.map(pd.DataFrame(dest_coords))
    else:
        st.info("No destination coordinates available to display.")

    # 📈 Flight Status Distribution
    st.markdown("## 📈 Flight Status Distribution")
    status_counts = {}
    for f in flights:
        status = f.get("status", "Unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    if status_counts:
        fig = px.pie(
            names=list(status_counts.keys()),
            values=list(status_counts.values()),
            title="Flight Status Distribution"
        )
        st.plotly_chart(fig)
    else:
        st.info("No status data available for flights.")

def dashboard_airport_capacity_vs_traffic():
    st.title("🛫 Airport Capacity vs. Actual Traffic")
    iata = st.text_input("Enter Airport IATA Code (e.g. JFK):").upper().strip()

    if iata:
        now = datetime.utcnow()
        end_time = now + timedelta(hours=5)
        scheduled = fetch_airport_timetable(iata, "departure")

        scheduled_filtered = []
        if isinstance(scheduled, list):
            for flight in scheduled:
                sched_time_str = flight.get("scheduledTime")
                if sched_time_str:
                    try:
                        sched_time = datetime.strptime(sched_time_str, "%Y-%m-%dT%H:%M:%SZ")
                        if now <= sched_time <= end_time:
                            scheduled_filtered.append(flight)
                    except Exception:
                        pass
        
        scheduled_count = len(scheduled_filtered)

        live = fetch_flights()
        live_count = 0
        if isinstance(live, list):
            for f in live:
                dep = f.get("departure")
                if dep and dep.get("iataCode") == iata:
                    live_count += 1
        
        df = pd.DataFrame({
            "Type": ["Scheduled (in 5 Hours)", "Actual (Live)"],
            "Flights": [scheduled_count, live_count]
        })
        
        st.write("Data for plotting:", df)  # Show dataframe values for debugging
        
        # Make sure Scheduled is first to show left, Actual second
        df = df.set_index("Type")
        st.bar_chart(df)# ================================


def dashboard_alternate_airports():
    st.title("🛬 Alternate Airport Suggestions")

    with st.form("alternate_airports_form"):
        lat = st.number_input("Current Latitude", format="%.6f")
        lng = st.number_input("Current Longitude", format="%.6f")
        distance = st.slider("Search Radius (km)", 50, 500, 100)
        submitted = st.form_submit_button("Find Nearby Airports")

    if not submitted:
        return

    API_KEY = "7b7ad2-eba700"
    nearby_url = f"https://aviation-edge.com/v2/public/nearby?key={API_KEY}&lat={lat}&lng={lng}&distance={distance}"

    with st.spinner("Finding nearby airports..."):
        try:
            airports = requests.get(nearby_url).json()
        except Exception as e:
            st.error(f"Failed to get nearby airports: {e}")
            return

    if not isinstance(airports, list) or not airports:
        st.warning("No airports found nearby.")
        return

    rows = []
    for airport in airports[:10]:  # Limit to 10 airports
        name = airport.get("nameAirport", "Unknown")
        dist = round(airport.get("distance", 0), 2)

        rows.append({
            "Airport": name,
            "Distance (km)": dist,
        })

    df = pd.DataFrame(rows)
    st.subheader("🛬 Suggested Alternate Airports")
    st.dataframe(df)

# Streamlit App Layout
# ================================
st.set_page_config(layout="wide", page_title="Aviation Dashboards", page_icon="✈️")
st.sidebar.title("✈️ Aviation Dashboard")
choice = st.sidebar.radio(
    "Choose a dashboard:",
    (
        "Real-Time Flight Map",
        "Airport Timetable",
        "Flight Delay Statistics",
        "Historical Traffic Trends",
        "Airlines",
        "Airport Capacity vs. Traffic",
	"Alternative Airports"
    )
)

if choice == "Real-Time Flight Map":
    dashboard_flight_map()
elif choice == "Airport Timetable":
    dashboard_airport_timetable()
elif choice == "Flight Delay Statistics":
    dashboard_flight_delay_statistics()
elif choice == "Historical Traffic Trends":
    dashboard_historical_traffic_trends()
elif choice == "Airlines":
    dashboard_airlines()
elif choice == "Airport Capacity vs. Traffic":
    dashboard_airport_capacity_vs_traffic()
elif choice == "Alternative Airports":
    dashboard_alternate_airports()
