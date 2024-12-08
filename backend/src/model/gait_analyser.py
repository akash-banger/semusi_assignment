import pandas as pd
import numpy as np
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from sqlalchemy import create_engine
from faker import Faker
import uuid
from src.constants import DB_CONNECTION_STRING

class GaitAnalyzer:
    def __init__(self):
        self.engine = create_engine(DB_CONNECTION_STRING)
        self.faker = Faker()
        self.scaler = StandardScaler()
        
    def extract_gait_features(self, person_data):
        """Extract comprehensive gait features from sensor data"""
        features = {}
        
        # Time domain features
        def calculate_sensor_features(data_x, data_y, data_z, prefix):
            magnitude = np.sqrt(data_x**2 + data_y**2 + data_z**2)
            return {
                f'{prefix}_mean_mag': np.mean(magnitude),
                f'{prefix}_std_mag': np.std(magnitude),
                f'{prefix}_max_mag': np.max(magnitude),
                f'{prefix}_min_mag': np.min(magnitude),
                f'{prefix}_kurtosis': stats.kurtosis(magnitude),
                f'{prefix}_skewness': stats.skew(magnitude),
                # Step time variability
                f'{prefix}_step_variability': self._calculate_step_variability(magnitude),
                # Symmetry features
                f'{prefix}_symmetry_index': self._calculate_symmetry_index(data_x, data_y, data_z)
            }

        # Extract features for each sensor location
        sensors = [
            ('right_foot', 'accelerometer_right_foot'),
            ('left_foot', 'accelerometer_left_foot'),
            ('right_shin', 'accelerometer_right_shin'),
            ('left_shin', 'accelerometer_left_shin'),
            ('right_thigh', 'accelerometer_right_thigh'),
            ('left_thigh', 'accelerometer_left_thigh')
        ]

        for location, prefix in sensors:
            x = person_data[f'{prefix}_x']
            y = person_data[f'{prefix}_y']
            z = person_data[f'{prefix}_z']
            features.update(calculate_sensor_features(x, y, z, location))

        # Add EMG features
        features.update({
            'emg_right_mean': np.mean(person_data['EMG_right']),
            'emg_left_mean': np.mean(person_data['EMG_left']),
            'emg_symmetry': self._calculate_emg_symmetry(
                person_data['EMG_right'], 
                person_data['EMG_left']
            )
        })

        return features

    def _calculate_step_variability(self, magnitude):
        """Calculate step-to-step variability using peak detection"""
        peaks, _ = signal.find_peaks(magnitude, distance=20)
        if len(peaks) < 2:
            return 0
        step_times = np.diff(peaks)
        return np.std(step_times) / np.mean(step_times)

    def _calculate_symmetry_index(self, data_x, data_y, data_z):
        """Calculate symmetry index between left and right sides"""
        magnitude = np.sqrt(data_x**2 + data_y**2 + data_z**2)
        half_len = len(magnitude) // 2
        return np.corrcoef(magnitude[:half_len], magnitude[half_len:])[0, 1]

    def _calculate_emg_symmetry(self, emg_right, emg_left):
        """Calculate EMG symmetry between right and left sides"""
        return np.corrcoef(emg_right, emg_left)[0, 1]

    def store_person_data(self, df, person_id):
        """Store person's gait data with features"""
        person_data = df[df['person_id'] == person_id].copy()
        
        # Generate random name (in real system, this would be actual person's name)
        person_name = self.faker.name()
        
        # Extract features
        features = self.extract_gait_features(person_data)
        
        # Create person profile
        profile = {
            'person_id': person_id,
            'name': person_name,
            'session_id': str(uuid.uuid4()),
            'features': features,
            'raw_data': person_data.to_dict('records')
        }
        
        # Store in database
        self._save_to_database(profile)
        
        return profile

    def identify_person(self, test_data):
        """Identify person based on their gait pattern"""
        # Extract features from test data
        test_features = self.extract_gait_features(test_data)
        
        # Get all stored profiles from database
        stored_profiles = self._get_stored_profiles()
        
        # Compare with stored profiles
        similarities = []
        for profile in stored_profiles:
            similarity = self._calculate_similarity(test_features, profile['features'])
            similarities.append({
                'person_id': profile['person_id'],
                'name': profile['name'],
                'similarity_score': similarity
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similarities

    def _calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        # Convert features to vectors
        feature_keys = sorted(features1.keys())
        vector1 = np.array([features1[k] for k in feature_keys])
        vector2 = np.array([features2[k] for k in feature_keys])
        
        # Normalize vectors
        vector1 = self.scaler.fit_transform(vector1.reshape(-1, 1)).ravel()
        vector2 = self.scaler.transform(vector2.reshape(-1, 1)).ravel()
        
        # Calculate cosine similarity
        similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        
        return similarity

    def _save_to_database(self, profile):
        """Save profile to database"""
        with self.engine.connect() as conn:
            # Store in persons table
            conn.execute("""
                INSERT INTO persons (person_id, name, session_id)
                VALUES (%(person_id)s, %(name)s, %(session_id)s)
            """, profile)
            
            # Store features
            conn.execute("""
                INSERT INTO gait_features (person_id, session_id, features)
                VALUES (%(person_id)s, %(session_id)s, %(features)s)
            """, {**profile, 'features': profile['features']})
            
            # Store raw data
            conn.execute("""
                INSERT INTO raw_gait_data (person_id, session_id, data)
                VALUES (%(person_id)s, %(session_id)s, %(raw_data)s)
            """, profile)

    def _get_stored_profiles(self):
        """Retrieve all stored profiles from database"""
        with self.engine.connect() as conn:
            return conn.execute("""
                SELECT p.person_id, p.name, p.session_id, f.features
                FROM persons p
                JOIN gait_features f ON p.person_id = f.person_id
            """).fetchall()