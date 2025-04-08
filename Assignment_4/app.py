# app.py

from flask import Flask, request, jsonify
from score import score
import joblib

# Load the best model saved during experiments
model_path = 'best_text_classifier.pkl'
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/score', methods=['POST'])
def get_score():
    content = request.json
    text = content['text']
    threshold = float(content['threshold'])

    # Score the text using the best model
    prediction, propensity = score(text, model, threshold)

    result = {
        'prediction': int(prediction),
        'propensity': propensity
    }

    print('Received Response')
    return jsonify(result)

if __name__ == '__main__':
    print('Running the app')
    app.run(debug=True, host='0.0.0.0')
