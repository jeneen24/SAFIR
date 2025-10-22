# weather_service.py
"""
Service for fetching weather data, bird observations, and calculating risks.
"""

import logging
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import math
import requests
from datetime import datetime
from sklearn.cluster import DBSCAN

from config import (
    OPENMETEO_URL, EBIRD_API_KEY_WEATHER, 
    CITIES_CSV, CORRIDOR_BUFFER_METERS
)
from utils import buffer_corridor_true_distance


HEADERS_BIRD = {'X-eBirdApiToken': EBIRD_API_KEY_WEATHER}
REQUEST_CACHE = requests.Session()


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