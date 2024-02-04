# Import Python tools for loading/navigating data
import pandas as pd   # Great for tables (google spreadsheets, microsoft excel, csv).
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import spacy
import wordcloud
import os # Good for navigating your computer's files
import sys
pd.options.mode.chained_assignment = None #suppress warnings

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
nltk.download('wordnet')
nltk.download('punkt')

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
!python -m spacy download en_core_web_md
import en_core_web_md
text_to_nlp = en_core_web_md.load()

from YelpModels import *

# Load the data.
yelp_full = pd.read_csv('yelp_data.csv')
yelp_full.head()
yelp = yelp_full


# Plot the word cloud - look at most prominent words
num_stars =  1#@param {type:"integer"}
this_star_text = ''
for t in yelp[yelp['stars'] == num_stars]['text'].values: # form field cell
    this_star_text += t + ' '

wordcloud = WordCloud()
wordcloud.generate_from_text(this_star_text)
plt.figure(figsize=(14,7))
plt.imshow(wordcloud, interpolation='bilinear')


# Classify positive and negative sentiment and convert stars column to either good or bad
def is_good_review(num_stars):
    if num_stars >= 4: 
        return True
    else:
        return False

kyelp['is_good_review'] = yelp['stars'].apply(is_good_review)
yelp.head()


# Tokenize
# Remove stopwords

X_text = yelp['text']
y = yelp['is_good_review']

def tokenize(text):
    clean_tokens = []
    for token in text_to_nlp(text):
        if (not token.is_stop) & (token.lemma_ != '-PRON-') & (not token.is_punct): # -PRON- is a special all inclusive "lemma" spaCy uses for any pronoun, we want to exclude these
            clean_tokens.append(token.lemma_)
    return clean_tokens

# Prepare Vocabulary
bow_transformer = CountVectorizer(analyzer=tokenize, max_features=800).fit(X_text)

# Use bag of words to prepare our data
X = bow_transformer.transform(X_text)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Train the model
model, train_countvect = train_model(X_train, y_train)

#Predict labels for test set
y_pred = predict (X_test, train_countvect, model)
print()
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print()



