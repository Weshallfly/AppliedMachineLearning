# score.py

from typing import Tuple
import numpy as np
import os
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Score a trained model on a text.

    Parameters:
    - text (str): The input text to be scored.
    - model (sklearn.base.BaseEstimator): The trained model.
    - threshold (float): The threshold value for classification.

    Returns:
    - prediction (bool): The predicted class based on the threshold.
    - propensity (float): The propensity score.
    """
    # Define the preprocessing pipeline
    preprocess_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1275)),
        # Add more preprocessing steps if needed
    ])
    script_directory = os.path.dirname(os.path.abspath(__file__))

    train = pd.read_csv(os.path.join(script_directory, 'train.csv')) 
    X,y = train['text'], train['spam']
    
    preprocess_pipeline.fit(X)

    # Preprocess the input text
    preprocessed_text = preprocess_pipeline.transform([text])

    # Perform prediction
    predicted_proba = model.predict_proba(preprocessed_text)[0]
    propensity_score = predicted_proba[1]  # Assuming the positive class is at index 1

    # Apply threshold for binary classification
    prediction = bool(propensity_score >= threshold)

    return prediction, float(propensity_score)
