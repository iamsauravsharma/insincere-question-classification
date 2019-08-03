import pandas as pd
from wordcloud import STOPWORDS
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt")

df = pd.read_csv("train.csv")
del df["qid"]
df.head()

question_text = df.question_text.str.cat(sep=" ")  # function to split text into word
tokens = word_tokenize(question_text)
vocabulary = set(tokens)
print(len(vocabulary))
frequency_dist = nltk.FreqDist(tokens)

stop_words = set(STOPWORDS)
tokens = [w for w in tokens if w not in stop_words]
frequency_dist = nltk.FreqDist(tokens)

X_train, X_test, y_train, y_test = train_test_split(
    df["question_text"], df["target"], test_size=0.10
)

vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)

"""## Logistic Regression"""
lr = LogisticRegression()
lr.fit(train_vectors, y_train)

prediction = lr.predict(test_vectors)
print("f1 score {}".format(f1_score(y_test, prediction)))
print("accuracy {}".format(accuracy_score(y_test, prediction)))
print("precision score {}".format(precision_score(y_test, prediction)))
print("recall_score {}".format(recall_score(y_test, prediction)))
cm_custom = confusion_matrix(y_test, prediction)
print(cm_custom)

"""## MultiNomial Naive Bayes Classification"""

mn = MultinomialNB()
mn.fit(train_vectors, y_train)

prediction = mn.predict(test_vectors)
print("f1 score {}".format(f1_score(y_test, prediction)))
print("accuracy {}".format(accuracy_score(y_test, prediction)))
print("precision score {}".format(precision_score(y_test, prediction)))
print("recall_score {}".format(recall_score(y_test, prediction)))
cm_custom = confusion_matrix(y_test, prediction)
print(cm_custom)

"""## K nearest neighbours"""

neigh = KNeighborsClassifier(n_neighbors=2)
neigh.fit(train_vectors, y_train)

prediction = neigh.predict(test_vectors)
print("f1 score {}".format(f1_score(y_test, prediction)))
print("accuracy {}".format(accuracy_score(y_test, prediction)))
print("precision score {}".format(precision_score(y_test, prediction)))
print("recall_score {}".format(recall_score(y_test, prediction)))
cm_custom = confusion_matrix(y_test, prediction)
print(cm_custom)
