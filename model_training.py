import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import re

# Function to preprocess the text data
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load your dataset
data = pd.read_csv(r'C:\Users\acer\OneDrive\Desktop\newsDetection\fake_news_data.csv', sep=',')  # Ensure the correct separator

# Strip any leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Check for NaN values and drop them
data.dropna(inplace=True)

# Encode labels
data['label'] = data['label'].map({'real': 1, 'fake': 0})

# Preprocess the text data
data['text'] = data['text'].apply(preprocess_text)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Vectorize the text
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer have been saved successfully.")
