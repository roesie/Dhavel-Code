
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Slearn for easy label encoding, textblob for senitment reference
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from textblob import TextBlob

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import StratifiedKFold

# NLP toolkits
import nltk
from nltk.corpus import stopwords
#nltk.download()
stopwords.words('english')
import string

# Importing the Dataset
data = pd.read_csv('sentiment.tsv', delimiter = '\t', quoting=3)

# Use a label encoder for the sentiment column. There's only two string values, 'pos' and 'neg', so the label encoder works wonders in a few lines of code.
le = LabelEncoder()
data['Sentiment'] = le.fit_transform(data['Sentiment'])
data['TextBlob Sentiment'] = (data['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity) / 2) + 0.5
data['TextBlob Sentiment'] = data['TextBlob Sentiment'].apply(lambda x: 1 if x > 0.5 else 0)
y = data['Sentiment'].as_matrix()

def clean_tweet(twt):
    twt = ''.join([char for char in twt if char not in string.punctuation])
    return [word for word in twt.split() if word.lower() not in stopwords.words('english')]

# Create bag of words
X = CountVectorizer(analyzer=clean_tweet).fit_transform(data['Tweet']).toarray()

# KFold cross-validation
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

# Creating Neural Network with Dropout layer since train model appears to overfit
for train, test in kfold.split(X, y):
    nlp_nn = Sequential()
    nlp_nn.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'tanh', input_shape = (X[train].shape[1],)))
    nlp_nn.add(Dropout(0.8))
    nlp_nn.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    nlp_nn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    nlp_nn.fit(X[train], y[train], batch_size = 30, epochs = 100, verbose=0)
    predictions = nlp_nn.predict(X[test])
    
    # Part 3 - Making predictions and evaluating the model
    predictions = list(map(lambda x: 1 if x > 0.5 else 0, predictions))
    print('Model Accuracy: {:.2f}%'.format(np.mean(predictions == y[test])*100))
    
print('TextBlob Accuracy: {:.2f}%'.format(np.mean(data['Sentiment'] == data['TextBlob Sentiment'])*100))
