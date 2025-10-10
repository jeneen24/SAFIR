from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Airport(db.Model):
    __tablename__ = 'airports'
    
    id = db.Column(db.Integer, primary_key=True)
    airport_id = db.Column(db.String, nullable=False, unique=True)
    name = db.Column(db.String, nullable=False)
    iata_code = db.Column(db.String, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    weather_data = db.relationship('WeatherData', backref='airport', lazy=True)

class City(db.Model):
    __tablename__ = 'cities'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    weather_data = db.relationship('WeatherData', backref='city', lazy=True)

class WeatherData(db.Model):
    __tablename__ = 'weather_data'
    
    id = db.Column(db.Integer, primary_key=True)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    precipitation = db.Column(db.Float)
    wind_speed = db.Column(db.Float)
    visibility = db.Column(db.Float)
    cloud_cover = db.Column(db.Float)
    timestamp = db.Column(db.DateTime)
    airport_id = db.Column(db.Integer, db.ForeignKey('airports.id'))
    city_id = db.Column(db.Integer, db.ForeignKey('cities.id'))

class BirdObservation(db.Model):
    __tablename__ = 'bird_observations'
    
    id = db.Column(db.Integer, primary_key=True)
    species = db.Column(db.String)
    count = db.Column(db.Integer)
    latitude = db.Column(db.Float)
    longitude = db.Column(db.Float)
    observation_time = db.Column(db.DateTime)