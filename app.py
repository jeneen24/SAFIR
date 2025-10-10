import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
from shapely.geometry import Point, LineString
import time
import math
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from datetime import datetime
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN
from math import radians, sin, cos, sqrt, atan2

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------------
# Global Configuration
# ---------------------------
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
EBIRD_API_KEY_WEATHER = "dbi6uafioq4u"
HEADERS_BIRD = {'X-eBirdApiToken': EBIRD_API_KEY_WEATHER}
REQUEST_CACHE = requests.Session()

AIRPORTS_CSV = 'data/airportsdatabase.csv'
CITIES_CSV = 'data/locations.csv'
CORRIDOR_BUFFER_METERS = 4 * 111320
EDGE_DIST_THRESHOLD = 555

# Aviation Stack API configuration (replace with your actual API key)
AVIATIONSTACK_API_KEY = "your_api_key_here"  # Get free key from https://aviationstack.com
AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1"


def haversine_distance(coord1: tuple, coord2: tuple) -> float:
    """Calculate great-circle distance between two points in kilometers."""
    lon1, lat1 = coord1
    lon2, lat2 = coord2
    lon1, lat1, lon2, lat2 = map(radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    earth_radius_km = 6371.0
    return earth_radius_km * c


def buffer_corridor_true_distance(dep_lon, dep_lat, arr_lon, arr_lat, buffer_meters):
    """Build a buffer around the great-circle line between two points."""
    route_line = LineString([(dep_lon, dep_lat), (arr_lon, arr_lat)])
    route_gdf = gpd.GeoDataFrame(geometry=[route_line], crs="EPSG:4326")
    mid_lon = (dep_lon + arr_lon) / 2.0
    mid_lat = (dep_lat + arr_lat) / 2.0
    aeqd_crs = (
        f"+proj=aeqd +lat_0={mid_lat:.6f} +lon_0={mid_lon:.6f} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    route_local = route_gdf.to_crs(aeqd_crs)
    corridor_local = route_local.buffer(buffer_meters).iloc[0]
    corridor_wgs84 = (
        gpd.GeoSeries([corridor_local], crs=aeqd_crs)
           .to_crs("EPSG:4326")
           .iloc[0]
    )
    return corridor_wgs84


# ---------------------------
# Weather & Bird Risk Service
# ---------------------------
class WeatherBirdsService:
    def __init__(self):
        logging.info("Initializing WeatherBirdsService.")

    def get_cities_in_corridor(self, dep_lon, dep_lat, arr_lon, arr_lat, buffer_distance_m=CORRIDOR_BUFFER_METERS):
        logging.info("Creating corridor from departure to arrival.")
        try:
            corridor = buffer_corridor_true_distance(
                dep_lon, dep_lat, arr_lon, arr_lat, buffer_distance_m)
        except Exception as e:
            logging.error("Error creating corridor: %s", e)
            raise Exception(f"Error creating corridor: {e}")

        logging.info(f"Loading cities from {CITIES_CSV}.")
        try:
            cities_df = pd.read_csv(CITIES_CSV, encoding="latin-1")
            if not {'longitudeAirport', 'latitudeAirport'}.issubset(cities_df.columns):
                raise Exception("Missing required columns in locations CSV.")
        except Exception as e:
            logging.error("Error loading cities CSV: %s", e)
            raise Exception(f"Error loading {CITIES_CSV}: {e}")

        logging.info("Creating GeoDataFrame for cities.")
        try:
            cities_gdf = gpd.GeoDataFrame(
                cities_df,
                geometry=[Point(xy) for xy in zip(cities_df['longitudeAirport'], cities_df['latitudeAirport'])],
                crs="EPSG:4326"
            )
        except Exception as e:
            logging.error("Error creating GeoDataFrame: %s", e)
            raise Exception(f"Error creating GeoDataFrame: {e}")

        cities_in_corridor = cities_gdf[cities_gdf.geometry.within(corridor)]
        if cities_in_corridor.empty:
            logging.warning("No cities found along the corridor.")
        else:
            logging.info(f"Found {len(cities_in_corridor)} cities in corridor.")
            cities_in_corridor.to_csv('data/filtered_cities.csv', index=False)
        return cities_in_corridor, corridor

    def cluster_city_points(self, cities_gdf, eps=40000, min_samples=2):
        logging.info("Clustering city points using DBSCAN.")
        mid_point = cities_gdf.unary_union.centroid
        aeqd_crs = (
            f"+proj=aeqd +lat_0={mid_point.y:.6f} +lon_0={mid_point.x:.6f} "
            "+datum=WGS84 +units=m +no_defs"
        )
        cities_local = cities_gdf.to_crs(aeqd_crs)
        points = np.vstack([cities_local.geometry.x, cities_local.geometry.y]).T
        dbscan = DBSCAN(eps=40000, min_samples=1)
        clusters = dbscan.fit_predict(points)
        unique_clusters = np.unique(clusters[clusters >= 0])
        num_clusters = len(unique_clusters)
        logging.info(f"DBSCAN found {num_clusters} clusters.")

        centroids = []
        for cl in unique_clusters:
            pts = points[clusters == cl]
            centroid_x, centroid_y = pts.mean(axis=0)
            centroid_geom = gpd.GeoSeries([Point(centroid_x, centroid_y)], crs=aeqd_crs).to_crs("EPSG:4326").iloc[0]
            centroids.append({'cluster': int(cl), 'centroid': (centroid_geom.x, centroid_geom.y)})
        
        centroids_df = pd.DataFrame([
            {'cluster': c['cluster'], 'longitude': c['centroid'][0], 'latitude': c['centroid'][1]} 
            for c in centroids
        ])
        centroids_df.to_csv('data/cluster_centroids.csv', index=False)
        return centroids

    def get_weather_data(self, lat: float, lon: float) -> dict:
        logging.info(f"Fetching weather data for ({lat}, {lon}).")
        params = {
            "latitude": lat,
            "longitude": lon,
            "current_weather": "true",
            "hourly": "temperature_80m,relativehumidity_2m,precipitation,cloudcover,windspeed_10m,windgusts_10m,visibility",
            "timezone": "auto",
            "forecast_days": 1
        }
        try:
            response = REQUEST_CACHE.get(OPENMETEO_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            current = data.get("current_weather", {})
            hourly = data.get("hourly", {})
            timestamp_str = current.get("time")
            if timestamp_str:
                if timestamp_str.endswith("Z"):
                    timestamp_str = timestamp_str.replace("Z", "+00:00")
                timestamp = datetime.fromisoformat(timestamp_str)
            else:
                timestamp = datetime.utcnow()

            def safe_first(key, default):
                arr = hourly.get(key, [])
                return arr[0] if arr and len(arr) > 0 else default

            weather = {
                "timestamp": timestamp,
                "temperature_2m": current.get("temperature"),
                "humidity": safe_first('relativehumidity_2m', 50),
                "precipitation": safe_first('precipitation', 0),
                "cloud_cover": safe_first('cloudcover', 0),
                "wind_speed": current.get("windspeed"),
                "wind_gusts": safe_first('windgusts_10m', 0),
                "visibility": safe_first('visibility', 10000),
                "weather_code": current.get("weathercode"),
                "temperature_80m": safe_first('temperature_80m', current.get("temperature")),
                "coordinates": (lat, lon)
            }
            return weather
        except Exception as e:
            logging.error(f"Weather API error for ({lat}, {lon}): {e}")
            return None

    def calculate_weather_risk(self, weather_data: dict) -> float:
        if not weather_data:
            return 0.0

        wind_speed_norm = (0, 25 * 0.514444)
        wind_gusts_norm = (0, 35 * 0.514444)
        precipitation_norm = (0, 10)
        visibility_norm = (500, 10000)
        cloud_cover_norm = (0, 90)
        temperature_ideal = (10, 30)
        humidity_ideal = (30, 70)

        def normalize(val, min_val, max_val):
            return max(0.0, min((val - min_val) / (max_val - min_val), 1.0))

        wind_risk = normalize(weather_data.get('wind_speed', 0), *wind_speed_norm)
        gust_risk = normalize(weather_data.get('wind_gusts', 0), *wind_gusts_norm)
        precip_risk = normalize(weather_data.get('precipitation', 0), *precipitation_norm)
        visibility_risk = 1 - normalize(weather_data.get('visibility', visibility_norm[1]), *visibility_norm)
        cloud_risk = normalize(weather_data.get('cloud_cover', 0), *cloud_cover_norm)

        severe_codes = [95, 96, 99]
        thunderstorm_risk = 1.0 if weather_data.get('weather_code') in severe_codes else 0.0
        icing_risk = 1.0 if (
            weather_data.get('temperature_80m', 10) < 0 and 
            weather_data.get('humidity', 50) > 75
        ) else 0.0

        temp_2m = weather_data.get('temperature_2m', 20)
        if temp_2m < temperature_ideal[0]:
            temperature_risk = normalize(temperature_ideal[0] - temp_2m, 0, temperature_ideal[0])
        elif temp_2m > temperature_ideal[1]:
            temperature_risk = normalize(temp_2m - temperature_ideal[1], 0, 15)
        else:
            temperature_risk = 0.0

        humidity = weather_data.get('humidity', 50)
        if humidity < humidity_ideal[0]:
            humidity_risk = normalize(humidity_ideal[0] - humidity, 0, humidity_ideal[0])
        elif humidity > humidity_ideal[1]:
            humidity_risk = normalize(humidity - humidity_ideal[1], 0, 30)
        else:
            humidity_risk = 0.0

        weights = {
            'wind_speed': 1.0,
            'wind_gusts': 1.8,
            'precipitation': 1.2,
            'visibility': 1.5,
            'cloud_cover': 0.6,
            'thunderstorm': 2.5,
            'icing': 2.0,
            'temperature': 0.8,
            'humidity': 0.5
        }

        total_risk = (
            wind_risk * weights['wind_speed'] +
            gust_risk * weights['wind_gusts'] +
            precip_risk * weights['precipitation'] +
            visibility_risk * weights['visibility'] +
            cloud_risk * weights['cloud_cover'] +
            thunderstorm_risk * weights['thunderstorm'] +
            icing_risk * weights['icing'] +
            temperature_risk * weights['temperature'] +
            humidity_risk * weights['humidity']
        )

        risk_score = min(max(total_risk, 0.0), 10.0)
        return risk_score

    def get_bird_observations(self, lat: float, lon: float, radius_km: float = 30) -> list:
        logging.info(f"Fetching bird observations for ({lat}, {lon}).")
        params = {
            'lat': lat,
            'lng': lon,
            'dist': radius_km,
            'maxResults': 500
        }
        try:
            response = REQUEST_CACHE.get("https://api.ebird.org/v2/data/obs/geo/recent",
                                         headers=HEADERS_BIRD, params=params, timeout=30)
            response.raise_for_status()
            observations = response.json()
            logging.info(f"Fetched {len(observations)} bird observations.")
            return observations
        except Exception as e:
            logging.error(f"Error retrieving bird observations: {e}")
            return []

    def calculate_bird_risk(self, observations: list, max_obs: int = 300) -> float:
        count = len(observations)
        scale = max_obs / 3.0
        risk = 10 * (1 - math.exp(-count / scale))
        return min(risk, 10)

    def process_corridor(self, dep_lon, dep_lat, arr_lon, arr_lat):
        try:
            cities_gdf, corridor = self.get_cities_in_corridor(dep_lon, dep_lat, arr_lon, arr_lat)
            centroids = self.cluster_city_points(cities_gdf)
        except Exception as e:
            raise Exception(f"Error in corridor processing: {e}")

        if cities_gdf.empty:
            raise Exception("No cities found along the corridor.")

        results = []
        for c in centroids:
            lon, lat = c['centroid']
            weather = self.get_weather_data(lat, lon)
            weather_risk = self.calculate_weather_risk(weather) if weather else None
            observations = self.get_bird_observations(lat, lon, radius_km=30)
            bird_risk = self.calculate_bird_risk(observations)
            
            results.append({
                'cluster': c['cluster'],
                'longitude': lon,
                'latitude': lat,
                'weather_risk': weather_risk,
                'bird_risk': bird_risk,
                'temperature_2m': weather['temperature_2m'] if weather else None,
                'precipitation': weather['precipitation'] if weather else None,
                'wind_speed': weather['wind_speed'] * 1.94384 if weather else None,
                'humidity': weather['humidity'] if weather else None,
                'bird_obs_count': len(observations)
            })

        final_df = pd.DataFrame(results)
        final_df.to_csv('data/final_cluster_results.csv', index=False)
        return final_df, corridor


# ---------------------------
# GNN Path Finding Service
# ---------------------------
class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_attr))
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        return x


class GNNPathFinder:
    def __init__(self, edge_threshold=EDGE_DIST_THRESHOLD):
        logging.info("Initializing GNNPathFinder.")
        self.edge_threshold = edge_threshold

    def build_graph_from_df(self, df: pd.DataFrame):
        logging.info("Building graph from DataFrame.")
        features = df[['longitude', 'latitude', 'weather_risk', 'bird_risk']].to_numpy(dtype=np.float32)
        coords = df[['longitude', 'latitude']].to_numpy(dtype=np.float32)
        edge_list = []
        edge_attr = []
        num_nodes = len(coords)
        
        for i in range(num_nodes):
            for j in range(i+1, num_nodes):
                dist = haversine_distance((coords[i][0], coords[i][1]), (coords[j][0], coords[j][1]))
                if dist <= self.edge_threshold:
                    risk_i = (df.iloc[i]['weather_risk'] or 0) + (df.iloc[i]['bird_risk'] or 0)
                    risk_j = (df.iloc[j]['weather_risk'] or 0) + (df.iloc[j]['bird_risk'] or 0)
                    risk = (risk_i + risk_j) / 2.0
                    edge_list.append([i, j])
                    edge_list.append([j, i])
                    edge_attr.append(risk)
                    edge_attr.append(risk)
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty(0, dtype=torch.float)
        
        data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)
        return data

    def build_networkx_graph(self, data: Data, df: pd.DataFrame, embeddings: np.ndarray):
        G = nx.Graph()
        for i, row in df.iterrows():
            G.add_node(i, longitude=row['longitude'], latitude=row['latitude'],
                       weather_risk=row['weather_risk'], bird_risk=row['bird_risk'])
        
        edge_index_np = data.edge_index.numpy()
        for idx in range(edge_index_np.shape[1]):
            i = edge_index_np[0, idx]
            j = edge_index_np[1, idx]
            risk = data.edge_attr[idx].item() if data.edge_attr.numel() > 0 else 0
            weight = haversine_distance(
                (G.nodes[i]['longitude'], G.nodes[i]['latitude']),
                (G.nodes[j]['longitude'], G.nodes[j]['latitude'])
            ) + risk
            weight += 0.5 * np.linalg.norm(embeddings[i] - embeddings[j])
            G.add_edge(i, j, weight=weight)
        return G

    def find_optimal_path(self, df: pd.DataFrame, start_coord: tuple, end_coord: tuple):
        data = self.build_graph_from_df(df)
        if data.edge_index.numel() == 0:
            raise Exception("Graph has no edges; cannot find path.")
        
        model = SimpleGCN(in_channels=4, hidden_channels=8, out_channels=4)
        model.eval()
        with torch.no_grad():
            embeddings = model(data.x, data.edge_index, data.edge_attr).numpy()
        
        G = self.build_networkx_graph(data, df, embeddings)

        def find_closest_node(target):
            return min(range(len(df)), key=lambda i: haversine_distance(
                (df.iloc[i]['longitude'], df.iloc[i]['latitude']), target))
        
        source_node = find_closest_node(start_coord)
        target_node = find_closest_node(end_coord)
        
        path = nx.dijkstra_path(G, source_node, target_node, weight='weight')
        total_cost = nx.dijkstra_path_length(G, source_node, target_node, weight='weight')
        return path, total_cost, df


# ---------------------------
# Flight Planner Integration
# ---------------------------
class FlightPlanner:
    def __init__(self, airports_csv=AIRPORTS_CSV):
        logging.info("Initializing FlightPlanner.")
        self.airports = pd.read_csv(airports_csv)
        required_cols = {'airportId', 'latitudeAirport', 'longitudeAirport', 'codeIataAirport', 'nameAirport'}
        if not required_cols.issubset(self.airports.columns):
            raise Exception("Missing required columns in airports CSV.")
        
        self.airports = self.airports.dropna(subset=['latitudeAirport', 'longitudeAirport'])
        self.airports['id'] = self.airports['airportId'].astype(str).str.strip()
        self.airports['iata_code'] = self.airports['codeIataAirport']
        self.airports['name'] = self.airports['nameAirport']
        self.airports['latitude_deg'] = self.airports['latitudeAirport']
        self.airports['longitude_deg'] = self.airports['longitudeAirport']
        
        self.weather_service = WeatherBirdsService()
        self.gnn_finder = GNNPathFinder()

    def plan_path(self, startId: str, endId: str):
        start = self.airports[self.airports['id'] == startId].iloc[0]
        end = self.airports[self.airports['id'] == endId].iloc[0]
        start_coord = (start['longitude_deg'], start['latitude_deg'])
        end_coord = (end['longitude_deg'], end['latitude_deg'])

        final_df, corridor = self.weather_service.process_corridor(
            dep_lon=start_coord[0], dep_lat=start_coord[1],
            arr_lon=end_coord[0], arr_lat=end_coord[1]
        )
        
        existing_labels = final_df['cluster'].astype(int).tolist()
        max_cluster = max(existing_labels) if existing_labels else -1
        start_label = max_cluster + 1
        end_label = max_cluster + 2

        def add_airport_node(df, coord, label):
            exists = any(math.isclose(row['longitude'], coord[0], abs_tol=1e-3) and 
                         math.isclose(row['latitude'], coord[1], abs_tol=1e-3) for _, row in df.iterrows())
            if not exists:
                weather = self.weather_service.get_weather_data(coord[1], coord[0])
                weather_risk = self.weather_service.calculate_weather_risk(weather) if weather else 0.0
                observations = self.weather_service.get_bird_observations(coord[1], coord[0], radius_km=30)
                bird_risk = self.weather_service.calculate_bird_risk(observations)
                return {
                    'cluster': label,
                    'longitude': coord[0],
                    'latitude': coord[1],
                    'weather_risk': weather_risk,
                    'bird_risk': bird_risk,
                    'temperature_2m': weather['temperature_2m'] if weather else None,
                    'precipitation': weather['precipitation'] if weather else None,
                    'wind_speed': weather['wind_speed'] if weather else None,
                    'humidity': weather['humidity'] if weather else None,
                    'bird_obs_count': len(observations)
                }
            return None

        start_node = add_airport_node(final_df, start_coord, start_label)
        if start_node:
            final_df = pd.concat([final_df, pd.DataFrame([start_node])], ignore_index=True)

        end_node = add_airport_node(final_df, end_coord, end_label)
        if end_node:
            final_df = pd.concat([final_df, pd.DataFrame([end_node])], ignore_index=True)

        path_indices, total_cost, node_df = self.gnn_finder.find_optimal_path(final_df, start_coord, end_coord)

        mid_lat = (start['latitude_deg'] + end['latitude_deg']) / 2
        mid_lon = (start['longitude_deg'] + end['longitude_deg']) / 2
        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=5)

        folium.Marker(
            location=[start['latitude_deg'], start['longitude_deg']],
            popup=f"Start: {start['name']}",
            icon=folium.Icon(color='blue', icon='plane', prefix='fa')
        ).add_to(m)
        
        folium.Marker(
            location=[end['latitude_deg'], end['longitude_deg']],
            popup=f"End: {end['name']}",
            icon=folium.Icon(color='darkblue', icon='flag', prefix='fa')
        ).add_to(m)

        if corridor:
            folium.GeoJson(
                data=gpd.GeoSeries(corridor).__geo_interface__,
                style_function=lambda x: {'fillColor': 'gray', 'color': 'gray', 'fillOpacity': 0.2}
            ).add_to(m)

        for idx, row in node_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color='lightblue',
                fill=True,
                fill_color='lightblue',
                fill_opacity=0.7,
                popup=f"Cluster: {row['cluster']}<br>Weather: {row['weather_risk']}<br>Bird: {row['bird_risk']}"
            ).add_to(m)

        path_coords = []
        for idx in path_indices:
            row = node_df.iloc[idx]
            path_coords.append((row['latitude'], row['longitude']))
            popup_text = (
                f"Cluster: {row['cluster']}<br>"
                f"Weather Risk: {row['weather_risk']}<br>"
                f"Bird Risk: {row['bird_risk']}<br>"
                f"Obs Count: {row['bird_obs_count']}"
            )
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=popup_text,
                icon=folium.Icon(color='red' if row['weather_risk'] and row['weather_risk'] > 5 else 'green')
            ).add_to(m)
        
        if path_coords:
            folium.PolyLine(path_coords, color='red', weight=5).add_to(m)
            heat_data = [[row['latitude'], row['longitude'], row['bird_risk']] 
                         for _, row in node_df.iterrows() if row['bird_risk'] is not None]
            if heat_data:
                HeatMap(heat_data, radius=25, blur=15, max_zoom=10).add_to(m)
        
        m.save('data/optimal_path_map.html')
        return m._repr_html_(), start['name'], end['name']

    def get_nearest_airports(self, lat: float, lng: float, distance_km: int = 90):
        """Find airports within specified distance from coordinates."""
        results = []
        for _, airport in self.airports.iterrows():
            dist = haversine_distance((lng, lat), (airport['longitude_deg'], airport['latitude_deg']))
            if dist <= distance_km:
                results.append({
                    'nameAirport': airport['name'],
                    'codeIataAirport': airport['iata_code'],
                    'nameCountry': airport.get('countryName', 'Unknown'),
                    'distance': round(dist * 1000),  # Convert to meters
                    'latitudeAirport': airport['latitude_deg'],
                    'longitudeAirport': airport['longitude_deg']
                })
        return sorted(results, key=lambda x: x['distance'])


# ---------------------------
# Flask App and Endpoints
# ---------------------------
app = Flask(__name__)
flight_planner = FlightPlanner()

# Example: create_tables.py
from flask import Flask
from models import db

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://user:password@localhost/dbname'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

with app.app_context():
    db.create_all()
    print("Tables created!")
@app.route('/')
def home():
    """Main application page."""
    return render_template('welcome.html')


@app.route('/welcome')
def welcome():
    """Welcome page before home."""
    return render_template('welcome.html')


@app.route('/get-airports')
def get_airports():
    """Return list of all airports."""
    try:
        airports_data = flight_planner.airports[['id', 'name', 'iata_code']].to_dict('records')
        return jsonify(airports_data)
    except Exception as e:
        logging.error("Error retrieving airports: %s", e)
        return jsonify({'error': str(e)}), 400


@app.route('/find-path', methods=['POST'])
def find_path():
    """Calculate optimal flight path between two airports."""
    data = request.json
    try:
        map_html, start_name, end_name = flight_planner.plan_path(data['startId'], data['endId'])
        return jsonify({
            'map_html': map_html,
            'start_name': start_name,
            'end_name': end_name
        })
    except Exception as e:
        logging.error("Error finding path: %s", e)
        return jsonify({'error': str(e)}), 400
@app.route('/nearest-airports', methods=['GET'])
def nearest_airports():
    """Get nearest airports to given coordinates."""
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
        distance = int(request.args.get('distance', 90))
        airports = flight_planner.get_nearest_airports(lat, lng, distance)
        return jsonify(airports)
    except Exception as e:
        logging.error("Error retrieving nearest airports: %s", e)
        return jsonify({'error': str(e)}), 400
@app.route('/data/<path:filename>')
def serve_data_file(filename):
    """Serve static data files."""
    return send_from_directory('data', filename)
@app.route('/static/<path:filename>')
def serve_static_file(filename):
    """Serve static files."""
    return send_from_directory('static', filename)
@app.route('/home')
def home_page():
    """Main application page."""
    return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)
