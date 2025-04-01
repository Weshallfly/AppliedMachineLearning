import pandas as pd
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib

# Load datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
validation_data = pd.read_csv('validation.csv')

# Assuming your dataset has 'text' as feature and 'spam' as the target column
X_train, y_train = train_data['text'], train_data['spam']
X_validation, y_validation = validation_data['text'], validation_data['spam']

# Convert text data to numerical format using suitable encoding (e.g., TF-IDF, CountVectorizer, etc.)
# Define the preprocessing pipeline
preprocess_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1275)),
    # Add more preprocessing steps if needed
])
    
X_train_transformed = preprocess_pipeline.fit_transform(X_train, y_train)
X_validation_transformed = preprocess_pipeline.fit_transform(X_validation, y_validation)

# Train XGBoost model
model = XGBClassifier()
model.fit(X_train_transformed, y_train)

# Evaluate model on validation set
accuracy = model.score(X_validation_transformed, y_validation)
print("Validation Accuracy:", accuracy)

# Save the trained model
joblib.dump(model, 'best_text_classifier.pkl')
