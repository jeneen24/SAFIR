# utils.py
"""
Utility functions for distance calculations and geometric operations.
"""

from math import radians, sin, cos, sqrt, atan2
from shapely.geometry import LineString
import geopandas as gpd


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