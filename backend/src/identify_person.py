import pandas as pd
from src.model.gait_analyser import GaitAnalyzer
import os

def identify_person():
    # Initialize analyzer
    analyzer = GaitAnalyzer()
    
    # Load and process test data
    test_data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'test_data.csv'))
    person_data = test_data.iloc[0]
    results = analyzer.identify_person(person_data)
    
    # Print top matches
    print("\nTop matches:")
    for match in results[:3]:
        print(f"Name: {match['name']}")
        print(f"Similarity Score: {match['similarity_score']:.2f}")
        print("---")

if __name__ == "__main__":
    identify_person() 