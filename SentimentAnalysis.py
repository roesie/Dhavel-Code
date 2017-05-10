
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Slearn for easy label encoding, textblob for senitment reference
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from textblob import TextBlob

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

# ML Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle

# NLP toolkits
import nltk, re
from nltk.corpus import stopwords, words
from nltk.stem import WordNetLemmatizer, PorterStemmer

#nltk.download()
import string

#%% Importing the Dataset
print('Importing Data...')
data = pd.read_csv('D:/Kaggle/Sentiment Analysis - Dhavel/sentiment.tsv', delimiter = '\t', quoting=3)
orig_len = len(data)

m_col = ['Sentiment', 'ID', 'Date', 'NA', 'Author', 'Tweet']
csvdata = pd.read_csv('D:/Kaggle/Sentiment Analysis - Dhavel/training.1600000.processed.noemoticon.csv', names=m_col, encoding='ISO-8859-1')
csvdata.drop(['ID','Date','NA','Author'], inplace=True, axis=1)
poscsvdata = csvdata[csvdata['Sentiment'] == 4]
negcsvdata = csvdata[csvdata['Sentiment'] == 0]

# Since sample sample is too large (800,000 samples) we choose a sample size better for our analysis
n_samples = 10000
poscsvdata = poscsvdata.iloc[np.random.choice(len(poscsvdata), size=n_samples)]
negcsvdata = negcsvdata.iloc[np.random.choice(len(negcsvdata), size=n_samples)]

# Clear memory of large csvdata variable
del csvdata

# Change values in positive senitment, 4 = pos, 0 = neg
poscsvdata['Sentiment'] = 1

# Use a label encoder for the sentiment column of original data
le = LabelEncoder()
data['Sentiment'] = le.fit_transform(data['Sentiment'])

# Concatenate 10000 samples into main dataframe
data = pd.concat([data, poscsvdata, negcsvdata], axis=0)
data = data.reset_index(drop=True).reset_index()
del poscsvdata, negcsvdata

#%% Define our tweet 'cleaner'
print('Cleaning Tweets...')

count = 1

def clean_tweet(twt):
    global count    
    # Remove stop words
    twt = [word for word in twt.split() if word.lower() not in stopwords.words('english')] 
    
    # Remove punctuation, join letters into string
    twt = ''.join([char for char in str(twt) if char not in string.punctuation ])    
    
    twt = re.sub('((www\S+)|(http\S+))', 'urlsite', twt)
    twt = re.sub(r'\d+', 'contnum', twt)
    
    if count % round(len(data['Tweet'])/100) == 0:
        sys.stdout.flush()
        sys.stdout.write('\rCompleted: {:.0f}%'.format(count/len(data['Tweet'])*100))
    count += 1
    return twt

data['Clean Tweet'] = data['Tweet'].apply(lambda x: clean_tweet(x))
data = shuffle(data)
X = data['Clean Tweet'].as_matrix()
y = data['Sentiment'].as_matrix()

#%% 
cv = CountVectorizer()
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.20, random_state=99)
Xtrain = cv.fit_transform(Xtrain).toarray()
Xtest = cv.transform(Xtest).toarray()
X_old = data[data['index'] < orig_len].sort_values('index')
y_old = X_old['Sentiment'].as_matrix()
X_old = X_old['Clean Tweet'].as_matrix()
X_old = cv.transform(X_old).toarray()

#%%
print('\nTraining Models...')

# Training the model
ann = Sequential()
ann.add(Dense(2500, kernel_initializer='truncated_normal', activation='relu', input_shape=(Xtrain.shape[1],)))
ann.add(Dropout(0.8))
ann.add(Dense(2500, kernel_initializer='truncated_normal', activation='relu'))
ann.add(Dropout(0.8))
ann.add(Dense(2500, kernel_initializer='truncated_normal', activation='relu'))
ann.add(Dropout(0.8))
ann.add(Dense(1, kernel_initializer='truncated_normal', activation='sigmoid'))
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(Xtrain, ytrain, batch_size=1500, epochs=50, verbose=1)

predictions1 = ann.evaluate(Xtest, ytest, verbose=0)[1]
predictions2 = ann.evaluate(X_old, y_old, verbose=0)[1]

#%%
print('Model Accuracy Overall: {:.1f}%, Model Accuracy Smallset: {:.1f}%'.format(predictions1*100, predictions2*100))
 #%% 
"""
print('Neural Network: {:.1f}%'.format(ann_score*100))      
print('TextBlob Accuracy: {:.2f}%'.format(np.mean(data['Sentiment'] == data['TextBlob Sentiment'])*100))

data['TextBlob Sentiment'] = (data['Tweet'].apply(lambda x: TextBlob(x).sentiment.polarity) / 2) + 0.5
data['TextBlob Sentiment'] = data['TextBlob Sentiment'].apply(lambda x: 1 if x > 0.5 else 0)
"""
