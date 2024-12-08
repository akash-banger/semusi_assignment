import pandas as pd
import numpy as np
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
from sqlalchemy import create_engine, text
from faker import Faker
import uuid
from src.constants import DB_CONNECTION_STRING
import json

class GaitAnalyzer:
    def __init__(self):
        self.engine = create_engine(DB_CONNECTION_STRING)
        self.faker = Faker()
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
    def extract_gait_features(self, person_data):
        """Extract comprehensive gait features from sensor data"""
        features = {}
        # Time domain features
        def calculate_sensor_features(data_x, data_y, data_z, prefix):
            # Convert scalar values to arrays if needed
            data_x = np.atleast_1d(data_x)
            data_y = np.atleast_1d(data_y)
            data_z = np.atleast_1d(data_z)
            
            # Check for NaN values in input data
            if np.any(np.isnan(data_x)) or np.any(np.isnan(data_y)) or np.any(np.isnan(data_z)):
                # Replace NaN with 0 for calculation
                data_x = np.nan_to_num(data_x, 0)
                data_y = np.nan_to_num(data_y, 0)
                data_z = np.nan_to_num(data_z, 0)
            
            magnitude = np.sqrt(data_x**2 + data_y**2 + data_z**2)
            features_dict = {
                f'{prefix}_mean_mag': float(np.mean(magnitude)),
                f'{prefix}_std_mag': float(np.std(magnitude)),
                f'{prefix}_max_mag': float(np.max(magnitude)),
                f'{prefix}_min_mag': float(np.min(magnitude)),
                f'{prefix}_kurtosis': float(stats.kurtosis(magnitude)),
                f'{prefix}_skewness': float(stats.skew(magnitude)),
                f'{prefix}_step_variability': float(self._calculate_step_variability(magnitude)),
                f'{prefix}_symmetry_index': float(self._calculate_symmetry_index(data_x, data_y, data_z))
            }
            
            # Verify no NaN values in features
            for key, value in features_dict.items():
                if np.isnan(value):
                    features_dict[key] = 0.0
            print(features_dict)
            return features_dict

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
            try:
                x = person_data[f'{prefix}_x']
                y = person_data[f'{prefix}_y']
                z = person_data[f'{prefix}_z']
                features.update(calculate_sensor_features(x, y, z, location))
            except KeyError as e:
                # Add zero values for missing sensors
                features.update({
                    f'{location}_mean_mag': 0.0,
                    f'{location}_std_mag': 0.0,
                    f'{location}_max_mag': 0.0,
                    f'{location}_min_mag': 0.0,
                    f'{location}_kurtosis': 0.0,
                    f'{location}_skewness': 0.0,
                    f'{location}_step_variability': 0.0,
                    f'{location}_symmetry_index': 0.0
                })

        # Add EMG features with NaN handling
        try:
            # Convert to arrays and handle NaN values
            emg_right = np.atleast_1d(np.asarray(person_data['EMG_right']))
            emg_left = np.atleast_1d(np.asarray(person_data['EMG_left']))
            
            # Handle NaN values
            emg_right = np.nan_to_num(emg_right, 0)
            emg_left = np.nan_to_num(emg_left, 0)
            
            features.update({
                'emg_right_mean': float(np.mean(emg_right)),
                'emg_left_mean': float(np.mean(emg_left)),
                'emg_symmetry': float(self._calculate_emg_symmetry(emg_right, emg_left))
            })
        except (KeyError, ValueError) as e:
            features.update({
                'emg_right_mean': 0.0,
                'emg_left_mean': 0.0,
                'emg_symmetry': 0.0
            })

        return features

    def _calculate_step_variability(self, magnitude):
        """Calculate step-to-step variability using peak detection"""
        # Ensure magnitude is a 1-D array
        magnitude = np.atleast_1d(magnitude)
        
        if len(magnitude) < 2:
            return 0
        
        peaks, _ = signal.find_peaks(magnitude, distance=20)
        if len(peaks) < 2:
            return 0
        step_times = np.diff(peaks)
        return np.std(step_times) / np.mean(step_times)

    def _calculate_symmetry_index(self, data_x, data_y, data_z):
        """Calculate symmetry index between left and right sides"""
        magnitude = np.sqrt(data_x**2 + data_y**2 + data_z**2)
        
        # Check if we have enough data
        if len(magnitude) < 2:
            return 0
        
        half_len = len(magnitude) // 2
        first_half = magnitude[:half_len]
        second_half = magnitude[half_len:half_len*2]  # Ensure equal length segments
        
        # Check if segments are non-empty and of equal length
        if len(first_half) < 2 or len(second_half) < 2 or len(first_half) != len(second_half):
            return 0
        
        try:
            return np.corrcoef(first_half, second_half)[0, 1]
        except:
            return 0

    def _calculate_emg_symmetry(self, emg_right, emg_left):
        """Calculate EMG symmetry between right and left sides"""
        # Convert inputs to arrays if they're scalars
        emg_right = np.atleast_1d(emg_right)
        emg_left = np.atleast_1d(emg_left)
        
        if len(emg_right) < 2 or len(emg_left) < 2:
            return 0
        
        try:
            return np.corrcoef(emg_right, emg_left)[0, 1]
        except:
            return 0

    def store_person_data(self, df, person_id):
        """Store person's gait data with features"""
        person_data = df[df['person_id'] == person_id].copy()
        
        # Generate random name (in real system, this would be actual person's name)
        person_name = self.faker.name()
        
        # Extract features
        features = self.extract_gait_features(person_data)
        
        # Create person profile with converted person_id to Python int
        profile = {
            'person_id': int(person_id),  # Convert numpy.int64 to Python int
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
        
        # Prepare all feature vectors for scaling
        all_feature_vectors = []
        feature_keys = sorted(test_features.keys())
        
        # Add test features and all stored features to the list
        all_feature_vectors.append([test_features[k] for k in feature_keys])
        for profile in stored_profiles:
            all_feature_vectors.append([profile['features'][k] for k in feature_keys])
        
        # Fit scaler on all data at once
        all_vectors_array = np.array(all_feature_vectors)
        normalized_vectors = self.scaler.fit_transform(all_vectors_array)
        
        # Get the normalized test vector (first in the array)
        normalized_test = normalized_vectors[0]
        
        # Compare with stored profiles
        similarities = []
        for idx, profile in enumerate(stored_profiles, 1):
            normalized_stored = normalized_vectors[idx]
            similarity = self._calculate_vector_similarity(normalized_test, normalized_stored)
            similarities.append({
                'person_id': profile['person_id'],
                'name': profile['name'],
                'similarity_score': similarity
            })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        print(similarities)
        
        return similarities

    def _calculate_vector_similarity(self, vector1, vector2):
        """Calculate normalized similarity score between two vectors"""
        try:
            # Calculate cosine similarity
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_similarity = np.dot(vector1, vector2) / (norm1 * norm2)
            
            # Convert from [-1, 1] range to [0, 1] range
            normalized_similarity = (cosine_similarity + 1) / 2
            
            if np.isnan(normalized_similarity):
                return 0.0
            
            return float(normalized_similarity)
        except Exception as e:
            return 0.0

    def _convert_to_serializable(self, obj):
        """Convert numpy types and nested structures to JSON-serializable format"""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Handle NaN values
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        elif isinstance(obj, (float, np.float64)) and np.isnan(obj):  # Only check isnan for numeric types
            return None
        return obj

    def _save_to_database(self, profile):
        """Save profile to database"""
        # Convert data to serializable format
        serializable_profile = {
            'person_id': profile['person_id'],
            'name': profile['name'],
            'session_id': profile['session_id'],
            'features': json.dumps(self._convert_to_serializable(profile['features'])),
            'raw_data': json.dumps(self._convert_to_serializable(profile['raw_data']))
        }
        
        with self.engine.connect() as conn:
            # Upsert into persons table
            conn.execute(
                text("""
                    INSERT INTO persons (person_id, name, session_id)
                    VALUES (:person_id, :name, :session_id)
                    ON CONFLICT (person_id) 
                    DO UPDATE SET 
                        name = EXCLUDED.name,
                        session_id = EXCLUDED.session_id
                """),
                serializable_profile
            )
            
            # Upsert features
            conn.execute(
                text("""
                    INSERT INTO gait_features (person_id, session_id, features)
                    VALUES (:person_id, :session_id, :features)
                    ON CONFLICT (person_id) 
                    DO UPDATE SET 
                        session_id = EXCLUDED.session_id,
                        features = EXCLUDED.features
                """),
                serializable_profile
            )
            
            # Upsert raw data - fixed the EXCLUDED.data reference
            conn.execute(
                text("""
                    INSERT INTO raw_gait_data (person_id, session_id, data)
                    VALUES (:person_id, :session_id, :raw_data)
                    ON CONFLICT (person_id) 
                    DO UPDATE SET 
                        session_id = EXCLUDED.session_id,
                        data = EXCLUDED.data
                """),
                serializable_profile
            )
            
            conn.commit()

    def _get_stored_profiles(self):
        """Retrieve all stored profiles from database"""
        with self.engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT p.person_id, p.name, p.session_id, f.features
                    FROM persons p
                    JOIN gait_features f ON p.person_id = f.person_id
                """)
            )
            
            # Convert the results to a list of dictionaries
            profiles = []
            for row in result:
                profile = {
                    'person_id': row.person_id,
                    'name': row.name,
                    'session_id': row.session_id,
                    'features': row.features if isinstance(row.features, dict) else json.loads(row.features)
                }
                profiles.append(profile)
                
            return profiles