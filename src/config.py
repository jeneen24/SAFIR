# config.py
"""
Configuration settings for the Flight Planner application.
"""

# API Configuration
OPENMETEO_URL = "https://api.open-meteo.com/v1/forecast"
EBIRD_API_KEY_WEATHER = "dbi6uafioq4u"
AVIATIONSTACK_API_KEY = "your_api_key_here"  # Get free key from https://aviationstack.com
AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1"

# File Paths
AIRPORTS_CSV = 'data/airportsdatabase.csv'
CITIES_CSV = 'data/locations.csv'

# Distance and Buffer Settings
CORRIDOR_BUFFER_METERS = 4 * 111320
EDGE_DIST_THRESHOLD = 555

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(levelname)s - %(message)s'
}