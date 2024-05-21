from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

# Load the dataset
data = pd.read_csv('yelp_sentiment.txt', header=None, sep='\t')
X_text = data[0]
y = data[1]

# Split the data into training and test sets
X_val, X_test, y_val, y_test = train_test_split(X_text, y, test_size=0.1, random_state=42)

import stanza

# Download the English model for Stanza
stanza.download('en')

# Define the Stanza pipeline with the sentiment analysis processor
nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')


def predict_sentiment_in_batch(texts):
    # Process texts in batch mode
    docs = [nlp(text) for text in texts]
    # Extract sentiment: 0 - negative, 1 - neutral, 2 - positive
    sentiments = [doc.sentences[0].sentiment for doc in docs]
    return sentiments

# Using the validation set to decide which class to assign to texts classified as neutral
# Define the mapping from neutral to the class you decide
neutral_to_class = {0: 0, 1 : 0, 2: 1}  # or {1: 'negative'} based on your decision

val_sentiments = predict_sentiment_in_batch(X_val.to_list())
transformed_val_sentiments = [neutral_to_class.get(sentiment, sentiment) for sentiment in val_sentiments]

print(y_val)
print(transformed_val_sentiments)
val_accuracy = accuracy_score(y_val, transformed_val_sentiments)
print(f'Validation Set Accuracy: {val_accuracy}')

# Apply the classifier to the test set in batch mode
test_sentiments = predict_sentiment_in_batch(X_test.to_list())
transformed_test_sentiments = [neutral_to_class.get(sentiment, sentiment) for sentiment in test_sentiments]

# Assuming y_test is numeric with 0 for negative and 1 for positive
test_accuracy = accuracy_score(y_test, transformed_test_sentiments)
print(f'Test Set Accuracy: {test_accuracy}')


# First
# If we assume that it 0: 0, 1 : 1, 2:1
# Neutral will be taken as positive result then
# Validation Set Accuracy: 0.95
# Test Set Accuracy: 0.93

# Second
# If we assume the 0:1, 1:0, 2:1
# Neutrall will be taken as negative then
# Validation Set Accuracy: 0.9577777777777777
# Test Set Accuracy: 0.95
