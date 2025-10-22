# flight_planner.py
"""
Main flight planning service that integrates weather, bird risk, and pathfinding.
"""

import logging
import pandas as pd
import math
import folium
import geopandas as gpd
from folium.plugins import HeatMap

from config import AIRPORTS_CSV
from utils import haversine_distance
from weather_service import WeatherBirdsService
from pathfinder import GNNPathFinder


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