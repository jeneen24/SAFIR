# src/app.py
"""
Flask application with routes for the Flight Planner.
"""

from asyncio import subprocess
import logging
import sys
import os

# Add parent directory to path to import from src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, redirect, render_template, request, jsonify, send_from_directory

from config import LOGGING_CONFIG
from flight_planner import FlightPlanner

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format']
)

# Initialize Flask app - specify template folder relative to project root
app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')

# Initialize flight planner
flight_planner = FlightPlanner()


@app.route('/')
def home():
    """Main application page."""
    return render_template('welcome.html')


@app.route('/welcome')
def welcome():
    """Welcome page before home."""
    return render_template('welcome.html')


@app.route('/home')
def home_page():
    """Main application page."""
    return render_template('home.html')

@app.route('/voice-assistant')
def voice_assistant():
    """Voice assistant page"""
    return render_template('voice_assistant.html')


@app.route('/dashboard')
def dashboard():
    """Launch Streamlit dashboard"""
    global streamlit_process

    # Stop any previous Streamlit instances
    os.system("pkill -f streamlit")

    try:
        # Start Streamlit on port 8501 in headless mode
        streamlit_process = subprocess.Popen(
            ['streamlit', 'run', 'dash.py', '--server.port=8501', '--server.headless=true'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"Error starting Streamlit: {e}")

    # Redirect user to Streamlit app
    return redirect('http://localhost:8501', code=302)

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
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    return send_from_directory(data_folder, filename)


@app.route('/static/<path:filename>')
def serve_static_file(filename):
    """Serve static files."""
    static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')
    return send_from_directory(static_folder, filename)


if __name__ == '__main__':
    app.run(debug=True)