from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Airport(db.Model):
    __tablename__ = 'airports'
    
    id = db.Column(db.Integer, primary_key=True)
    airport_id = db.Column(db.String(50), unique=True, nullable=False)
    name = db.Column(db.String(255), nullable=False)
    iata_code = db.Column(db.String(10), nullable=False)
    latitude = db.Column(db.Numeric(9, 6), nullable=False)
    longitude = db.Column(db.Numeric(9, 6), nullable=False)
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)

class City(db.Model):
    __tablename__ = 'cities'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255), nullable=False)
    longitude = db.Column(db.Numeric(9, 6), nullable=False)
    latitude = db.Column(db.Numeric(9, 6), nullable=False)
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)
    weather_data = db.relationship('WeatherData', backref='city', lazy=True)
    bird_observations = db.relationship('BirdObservation', backref='city', lazy=True)

class WeatherData(db.Model):
    __tablename__ = 'weather_data'
    
    id = db.Column(db.Integer, primary_key=True)
    city_id = db.Column(db.Integer, db.ForeignKey('cities.id', ondelete='CASCADE'))
    timestamp = db.Column(db.DateTime, nullable=False)
    temperature = db.Column(db.Numeric(5, 2))
    humidity = db.Column(db.Numeric(5, 2))
    precipitation = db.Column(db.Numeric(5, 2))
    wind_speed = db.Column(db.Numeric(5, 2))
    visibility = db.Column(db.Numeric(5, 2))
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)

class BirdObservation(db.Model):
    __tablename__ = 'bird_observations'
    
    id = db.Column(db.Integer, primary_key=True)
    city_id = db.Column(db.Integer, db.ForeignKey('cities.id', ondelete='CASCADE'))
    observation_date = db.Column(db.DateTime, nullable=False)
    species = db.Column(db.String(255), nullable=False)
    count = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime)