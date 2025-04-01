# test.py

import pytest
import joblib
import os
import requests
import subprocess
import time
from score import score

# Get the directory where the current script resides
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the pickle file
pickle_file_path = os.path.join(script_directory, 'best_text_classifier.pkl')

# Load the best model saved during experiments
best_model = joblib.load(pickle_file_path)

def test_score():
    # Smoke test: Check if the function produces some output without crashing
    text = "Test text"
    threshold = 0.5
    prediction, propensity_score = score(text, best_model, threshold)
    assert prediction is not None
    assert propensity_score is not None
    print('Smoke Test: Success')

    # Format test: Check if the output formats/types are as expected
    assert isinstance(prediction, bool)
    assert isinstance(propensity_score, float)
    print('Format Test: Success')

    # Prediction value test: Check if the prediction value is 0 or 1
    assert prediction in [0, 1]
    print('Prediction Value Test: Success')

    # Propensity score range test: Check if the propensity score is between 0 and 1
    assert 0 <= propensity_score <= 1
    print('Propensity Score Test: Success')

    # Threshold tests
    # If the threshold is set to 0, the prediction should always be 1
    prediction, _ = score(text, best_model, threshold=0)
    assert prediction == 1
    print('0 Threshold Test: Success')

    # If the threshold is set to 1, the prediction should always be 0
    prediction, _ = score(text, best_model, threshold=1)
    assert prediction == 0
    print('1 Threshold Test: Success')

    # Obvious spam input text test: Prediction should be 1
    spam_text = "Congratulations! You've won a free trip to Hawaii with our special travel insurance and a bonus cash prize of $1000! Claim your reward now by clicking the link below. Don't miss out on this amazing offer!"
    prediction, propensity_score = score(spam_text, best_model, threshold=0.5)
    assert prediction == 1
    print('Spam Test: Success')

    # Obvious non-spam input text test: Prediction should be 0
    non_spam_text = "Hello, how are you?"
    prediction, propensity_score = score(non_spam_text, best_model, threshold=0.5)
    assert prediction == 0
    print('Non-Spam Test: Success')

    print('All test cases passed successfully.')

# Integration test function
def test_flask(text, threshold):
    # Start the Flask app using command line
    dir = os.path.dirname(os.path.abspath(__file__))
    filename = "app.py"
    full_path = os.path.join(dir, filename)
    flask_process = subprocess.Popen(["python3", full_path], stdout=subprocess.PIPE)
    
    # Wait for the server to start
    time.sleep(2)

    # Test the response from the localhost endpoint
    payload = {
        'text': text,
        'threshold': threshold
    }
    response = requests.post('http://127.0.0.1:5000/score', json=payload)
    data = response.json()
    prediction = data['prediction']
    propensity = data['propensity']
    # Check if the response contains prediction and propensity
    assert 'prediction' in data
    assert 'propensity' in data

    prediction_str = "spam" if prediction else "non-spam"

    # Print the result
    print(f'The text to be tested was "{text}"')
    print(f'It was classified as {prediction_str} with score {propensity}')
    
    # Stop the Flask app using command line
    print('Closing the app')
    os.system("pkill -f app.py")


if __name__ == "__main__":
    
    # Test the Score function
    test_score()
    
    # Testing the app
    print('Launching App')
    text = "URGENT: Exclusive Insurance Offer! Act Now to Secure Your Future! Limited Time Only: Claim Your Policy Before It's Too Late! Don't Miss Out on This Life-Saving Opportunity!"
    threshold = 0.5
    test_flask(text, threshold)
        
    
