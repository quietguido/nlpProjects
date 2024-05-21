import pandas as pd
from sklearn import model_selection as skm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import fasttext
import numpy as np

# Load the dataset
data = pd.read_csv('yelp_sentiment.txt', header=None, sep='\t')

# Separate the text and the labels
X_text = data[0]
y = data[1]

# Split the data into training and test sets
X_text_train, X_text_test, y_train, y_test = skm.train_test_split(X_text, y, test_size=0.1, stratify=y, random_state=20)

# Load the pre-trained fastText model
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English
ft_model = fasttext.load_model('cc.en.300.bin')

# Function to convert sentences to fastText vectors
def sentences_to_vectors(sentences, model):
    vectors = [model.get_sentence_vector(sentence) for sentence in sentences]
    return np.array(vectors)

# Convert sentences to vectors
X_train_vectors = sentences_to_vectors(X_text_train, ft_model)
X_test_vectors = sentences_to_vectors(X_text_test, ft_model)

# Initialize and train the logistic regression model using fastText vectors
log_reg_ft = LogisticRegression(max_iter=1000)
log_reg_ft.fit(X_train_vectors, y_train)

# Predict on the test set and evaluate
y_pred_ft = log_reg_ft.predict(X_test_vectors)
print(classification_report(y_test, y_pred_ft))
print("Accuracy (fastText):", accuracy_score(y_test, y_pred_ft))

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit the vectorizer on the training data and transform the training set
X_train_tfidf = tfidf_vectorizer.fit_transform(X_text_train)

# Transform the test set
X_test_tfidf = tfidf_vectorizer.transform(X_text_test)

# Train the logistic regression model using TF-IDF vectors
log_reg_tfidf = LogisticRegression(max_iter=1000)
log_reg_tfidf.fit(X_train_tfidf, y_train)

# Predict on the test set and evaluate
y_pred_tfidf = log_reg_tfidf.predict(X_test_tfidf)
print(classification_report(y_test, y_pred_tfidf))
print("Accuracy (TF-IDF):", accuracy_score(y_test, y_pred_tfidf))


'''

              precision    recall  f1-score   support

           0       0.70      0.76      0.73        50
           1       0.74      0.68      0.71        50

    accuracy                           0.72       100
   macro avg       0.72      0.72      0.72       100
weighted avg       0.72      0.72      0.72       100

Accuracy (fastText): 0.72
              precision    recall  f1-score   support

           0       0.84      0.86      0.85        50
           1       0.86      0.84      0.85        50

    accuracy                           0.85       100
   macro avg       0.85      0.85      0.85       100
weighted avg       0.85      0.85      0.85       100

Accuracy (TF-IDF): 0.85

'''