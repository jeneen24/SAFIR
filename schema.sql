CREATE TABLE airports (
    id SERIAL PRIMARY KEY,
    airport_id VARCHAR(50) NOT NULL UNIQUE,
    name VARCHAR(255) NOT NULL,
    iata_code VARCHAR(10) NOT NULL,
    latitude DECIMAL(9, 6) NOT NULL,
    longitude DECIMAL(9, 6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE cities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    longitude DECIMAL(9, 6) NOT NULL,
    latitude DECIMAL(9, 6) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE weather_data (
    id SERIAL PRIMARY KEY,
    city_id INTEGER REFERENCES cities(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    temperature DECIMAL(5, 2),
    humidity DECIMAL(5, 2),
    precipitation DECIMAL(5, 2),
    wind_speed DECIMAL(5, 2),
    visibility DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE bird_observations (
    id SERIAL PRIMARY KEY,
    city_id INTEGER REFERENCES cities(id) ON DELETE CASCADE,
    observation_date TIMESTAMP NOT NULL,
    species VARCHAR(255) NOT NULL,
    count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);