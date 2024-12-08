-- Drop existing tables if they exist
DROP TABLE IF EXISTS raw_gait_data;
DROP TABLE IF EXISTS gait_features;
DROP TABLE IF EXISTS persons;

-- Create persons table with primary key
CREATE TABLE persons (
    person_id INTEGER PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    session_id UUID NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create gait_features table with primary key
CREATE TABLE gait_features (
    person_id INTEGER PRIMARY KEY REFERENCES persons(person_id),
    session_id UUID NOT NULL,
    features JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create raw_gait_data table with primary key
CREATE TABLE raw_gait_data (
    person_id INTEGER PRIMARY KEY REFERENCES persons(person_id),
    session_id UUID NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

