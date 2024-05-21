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
X_train, X_test, y_train, y_test = train_test_split(X_text, y, test_size=0.1, random_state=42)

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode the sentences
X_train_encoded = model.encode(X_train.to_list(), show_progress_bar=True)
X_test_encoded = model.encode(X_test.to_list(), show_progress_bar=True)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_encoded, y_train)

y_pred = classifier.predict(X_test_encoded)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# 
# File All_miniLM.py

#               precision    recall  f1-score   support

#            0       0.78      0.95      0.86        44
#            1       0.96      0.79      0.86        56

#     accuracy                           0.86       100
#    macro avg       0.87      0.87      0.86       100
# weighted avg       0.88      0.86      0.86       100

# Accuracy: 0.86

# The result for the allMiniLM-L6-v2 is 0.86. But there is slight confusion from recall part in negative data set. Need to investigate why is so high in the negative part.
