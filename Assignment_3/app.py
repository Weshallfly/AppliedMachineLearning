# app.py

from flask import Flask, request, jsonify
from score import score
import joblib
import os

# Get the directory where the current script resides
script_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the pickle file
pickle_file_path = os.path.join(script_directory, 'best_text_classifier.pkl')

# Load the best model
model = joblib.load(pickle_file_path)

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def get_score():
    data = request.json
    text = data['text']
    threshold = float(data['threshold'])
    
    # Score the text using the best model
    prediction, propensity = score(text, model, threshold)
    
    response = {
        'prediction': int(prediction),
        'propensity': propensity
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
