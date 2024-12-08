import pandas as pd
from src.model.gait_analyser import GaitAnalyzer
import os

def main():
    # Initialize analyzer
    analyzer = GaitAnalyzer()
    
    # Load training data
    training_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'src', 'dataset', 'cleaned_data.csv'))
    
    # Store data for each person
    for person_id in training_data['person_id'].unique():
        profile = analyzer.store_person_data(training_data, person_id)
        print(f"Stored profile for person {profile['name']} (ID: {person_id})")
    
    # Example: Identify a person from test data
    test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'src', 'dataset', 'test_data.csv'))
    results = analyzer.identify_person(test_data)
    
    # Print top matches
    print("\nTop matches:")
    for match in results[:3]:
        print(f"Name: {match['name']}")
        print(f"Similarity Score: {match['similarity_score']:.2f}")
        print("---")

if __name__ == "__main__":
    main()