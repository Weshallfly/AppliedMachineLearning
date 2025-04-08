# score.py

from typing import Tuple
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Score a trained model on a text input.

    Parameters:
    - text (str): The input text to be scored.
    - model (BaseEstimator): The trained classification model.
    - threshold (float): Threshold value for making binary decision.

    Returns:
    - prediction (bool): Binary prediction based on threshold.
    - propensity (float): Propensity score for the positive class.
    """

    # Create the preprocessing pipeline
    preprocess_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=2000)),
        # Additional preprocessing steps can be added here
    ])

    # Load training data for fitting the vectorizer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = pd.read_csv(os.path.join(current_dir, 'train.csv'))
    X, y = train_data['text'], train_data['spam']

    # Fit the preprocessing pipeline
    preprocess_pipeline.fit(X)

    # Transform the input text
    transformed_text = preprocess_pipeline.transform([text])

    # Get model's probability predictions
    proba = model.predict_proba(transformed_text)[0]
    propensity_score = proba[1]  # Positive class assumed to be at index 1

    # Classify based on threshold
    prediction = bool(propensity_score >= threshold)

    return prediction, float(propensity_score)