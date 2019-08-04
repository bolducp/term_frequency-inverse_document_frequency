import tf_idf as tf_idf

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn import datasets
from sklearn.svm import l1_min_c
from sklearn.metrics import accuracy_score

import pandas as pd

tweets = pd.read_csv('tweets.csv')
documents = tweets.text.values
labels = tweets.handle.astype('category').cat.codes.values
train_docs = documents[:4000]
test_docs = documents[4000:4500]

parsed_train_documents = [tf_idf.parse(doc) for doc in train_docs]
parsed_test_documents = [tf_idf.parse(doc) for doc in test_docs]
vocab_map = tf_idf.create_vocab_map(parsed_train_documents)

def generate_tf_idfs(parsed_documents, vocab_map):
    idf = tf_idf.create_idf_array(vocab_map, parsed_documents)
    return np.array([tf_idf.calculate_tf_idf(vocab_map, tf_idf.create_word_freq_array(tf_idf.parse(document), vocab_map), idf) for document in parsed_documents])

x_train = generate_tf_idfs(parsed_train_documents, vocab_map)
y_train = np.array(labels[:4000])

x_test = generate_tf_idfs(parsed_test_documents, vocab_map)
y_test = np.array(labels[4000:4500])

clf = linear_model.LogisticRegression(penalty='l1', solver='saga', tol=1e-6, max_iter=int(1e6), warm_start=True)
clf.fit(x_train, y_train)

predictions = clf.predict(x_test)
# training on the first 4,000 tweets yields an accuracy of 0.806
print(accuracy_score(y_test, predictions))
