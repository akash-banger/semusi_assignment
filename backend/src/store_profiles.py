import pandas as pd
from src.model.gait_analyser import GaitAnalyzer
import os

def store_profiles():
    # Initialize analyzer
    analyzer = GaitAnalyzer()
    
    # Load training data
    training_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'cleaned_data.csv'))
    
    # Store data for each person
    for person_id in training_data['person_id'].unique():
        profile = analyzer.store_person_data(training_data, person_id)
        print(f"Stored profile for person {profile['name']} (ID: {person_id})")

if __name__ == "__main__":
    store_profiles() 