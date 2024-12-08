-- Persons table
CREATE TABLE persons (
    id SERIAL PRIMARY KEY,
    person_id INTEGER UNIQUE,
    name VARCHAR(100),
    session_id UUID,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Gait features table
CREATE TABLE gait_features (
    id SERIAL PRIMARY KEY,
    person_id INTEGER REFERENCES persons(person_id),
    session_id UUID,
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Raw data table
CREATE TABLE raw_gait_data (
    id SERIAL PRIMARY KEY,
    person_id INTEGER REFERENCES persons(person_id),
    session_id UUID,
    data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);