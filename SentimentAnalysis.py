
# Importing the Libraries
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# Slearn for easy label encoding, textblob for senitment reference
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# ML Models
from sklearn.utils import shuffle

# NLP toolkits
import nltk, re
from nltk.stem import WordNetLemmatizer, PorterStemmer

#nltk.download()
import string

#%% Importing the Dataset
print('Importing Data...')
data_original = pd.read_csv('D:/Kaggle/Sentiment Analysis - Dhavel/sentiment.tsv', delimiter = '\t', quoting=3)
data_original_len = len(data_original)

# Bootstrapping
col_name = ['Sentiment', 'ID', 'Date', 'NA', 'Author', 'Tweet']
data_resample = pd.read_csv('D:/Kaggle/Sentiment Analysis - Dhavel/training.1600000.processed.noemoticon.csv', names=col_name, encoding='ISO-8859-1')
data_resample.drop(['ID','Date','NA','Author'], inplace=True, axis=1)
pos_resample = data_resample[data_resample['Sentiment'] == 4]
neg_resample = data_resample[data_resample['Sentiment'] == 0]

n_resamples = 15000
pos_resample = pos_resample.iloc[np.random.choice(len(pos_resample), size=n_resamples)]
neg_resample = neg_resample.iloc[np.random.choice(len(neg_resample), size=n_resamples)]

# Clear memory of large csvdata variable
del data_resample

# Change values in positive senitment, 4 = pos, 0 = neg
pos_resample['Sentiment'] = 1

# Use a label encoder for the sentiment column of original data
le = LabelEncoder()
data_original['Sentiment'] = le.fit_transform(data_original['Sentiment'])

# Concatenate 10000 samples into main dataframe
data_resample = pd.concat([pos_resample, neg_resample], axis=0)
data_resample = data_resample.reset_index(drop=True).reset_index()
del pos_resample, neg_resample

#%% Define our tweet 'cleaner'
print('Cleaning Tweets...',) # Maybe look into a spellchecker??
count = 1
def clean_tweet(tweet, count_total):
    global count    
    # Remove '@' people's names
    tweet = [word for word in tweet.split() if not word.startswith('@')]
    tweet = [char for char in str(tweet) if char not in string.punctuation]
    tweet = ''.join(tweet)
    tweet = re.sub('((www\S+)|(http\S+))', 'urlsite', tweet)
    tweet = re.sub(r'\d+', 'num', tweet)
    
    sys.stdout.flush()
    sys.stdout.write('\r{}/{}'.format(count, count_total))
        #sys.stdout.write('\rCompleted: {:.0f}%'.format(count/len(data['Tweet'])*100))
    count += 1
    return tweet

data_original['Clean Tweet'] = data_original['Tweet'].apply(lambda x: clean_tweet(x, len(data_original)))
data_resample['Clean Tweet'] = data_resample['Tweet'].apply(lambda x: clean_tweet(x, len(data_resample) + len(data_original)))
data_resample = shuffle(data_resample)
X_resample = data_resample['Clean Tweet']
y_resample = data_resample['Sentiment'].as_matrix()

#%% 
# Using original data set as additional test set
X_original = data_original['Clean Tweet']
y_original = data_original['Sentiment'].as_matrix()

#%%
print('\nTraining Models...')
t0 = datetime.now()
cv = CountVectorizer()

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=99)

n_classifier = 2

for train, valid in kfold.split(X_resample, y_resample):
    # Separate our samples
    X_train = X_resample[train]
    y_train = y_resample[train]
    X_valid = X_resample[valid]
    y_valid = y_resample[valid]
    
    # Get our bag of words
    X_train = cv.fit_transform(X_train).toarray()
    X_valid = cv.transform(X_valid).toarray()
    X_test = cv.transform(X_original).toarray()
    y_test = y_original
    
    if n_classifier == 1:
        # RandomForest
        model = RandomForestClassifier(n_estimators=100, random_state=99)
        model = model.fit(X_train, y_train)
        
    elif n_classifier == 2:   
        # Multinomial NB
        model = MultinomialNB()
        model = model.fit(X_train, y_train)
        
    elif n_classifier == 3:      
        # 3 layer Neural Network
        model = Sequential()
        model.add(Dense(2500, kernel_initializer='truncated_normal', activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(1, kernel_initializer='truncated_normal', activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        model.fit(X_train, y_train, batch_size=1000, epochs=5, verbose=1)
        
    predictions_resample = model.predict(X_valid)
    predictions_original = model.predict(X_test)
        
    fpr_resample, tpr_resample, _ = roc_curve(y_valid, model.predict_proba(X_valid)[:,1])
    roc_auc_resample = auc(fpr_resample, tpr_resample)
    
    fpr_original, tpr_original, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
    roc_auc_original = auc(fpr_original, tpr_original)
    
    print('Resampling AUC Score: {:.3f}, Original AUC Score: {:.3f}'.format(roc_auc_resample, roc_auc_original))
    print('Resampling Accuracy: {:.1f}%, Original Accuracy: {:.1f}%'.format(np.mean(predictions_resample == y_valid)*100, np.mean(predictions_original == y_test)*100))   
#%%
print('Total Elapsed Time: {}'.format(datetime.now()-t0))

 #%% 

