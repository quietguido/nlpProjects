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
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Encode the sentences
X_train_encoded = model.encode(X_train.to_list(), show_progress_bar=True)
X_test_encoded = model.encode(X_test.to_list(), show_progress_bar=True)

classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_encoded, y_train)

y_pred = classifier.predict(X_test_encoded)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


# File paraphare.py

#               precision    recall  f1-score   support

#            0       0.75      0.93      0.83        44
#            1       0.93      0.75      0.83        56

#     accuracy                           0.83       100
#    macro avg       0.84      0.84      0.83       100
# weighted avg       0.85      0.83      0.83       100

# Accuracy: 0.83

# This model is one of the smallest and fastests ‘paraphrase-MiniLM-L3-v2’
# than all-MiniLM-L6-v2, which might be more efficient for applications where inference speed is critical. Also I did not like waiting too much to test my theories that is why I stated with the fastest. Result if accuracy 0.83. 


# Also he model is fine-tuned for paraphrasing tasks in Yelp reviews. Which means it is good at saying negative or positive sentences in different ways. Which I thought will give some advantage from that previous model. But difference is negligible. 
