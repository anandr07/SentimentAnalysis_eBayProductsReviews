import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
import joblib
import sqlite3
from bs4 import BeautifulSoup
import re

# Define the stopwords set
stopwords = set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def clean_text(sentence):
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'html.parser').get_text()
    sentence = decontracted(sentence)
    sentence = re.sub(r"\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)
    sentence = ' '.join(e.lower() for e in sentence.split() if e.lower() not in stopwords)
    return sentence.strip()

def predict(string):
    clf = joblib.load('Model(SGDC).pkl')
    count_vect = joblib.load('CountVectorizer.pkl')
    review_text = clean_text(string)
    test_vect = count_vect.transform(([review_text]))
    pred = clf.predict(test_vect)
    prediction = "Positive" if pred[0] else "Negative"
    return prediction

def partition(x):
    if x < 3:
        return 0  # Negative
    elif x == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

# Load data from SQLite database
# con = sqlite3.connect('data/database.sqlite')
# filtered_data = pd.read_sql_query("SELECT * FROM Reviews WHERE Score != 3 LIMIT 10000", con)
filtered_data = pd.read_csv('ebay_reviews.csv')

# Map scores to sentiment classes
filtered_data['Score'] = filtered_data['Score'].map(partition)

# Sort and remove duplicates
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
final = sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)
# final = final.sort_values('Time', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
# final = final[final.HelpfulnessNumerator <= final.HelpfulnessDenominator]

# Preprocess text data
preprocessed_reviews = []
for sentence in final['Text'].values:
    preprocessed_reviews.append(clean_text(sentence))

# Fit CountVectorizer and transform data
count_vect = CountVectorizer()
count_vect.fit(preprocessed_reviews)
joblib.dump(count_vect, 'CountVectorizer.pkl')
X = count_vect.transform(preprocessed_reviews)
Y = final['Score'].values

# Train linear SVM classifier
clf = linear_model.SGDClassifier(max_iter=1000, tol=1e-3, eta0=0.1, alpha=0.001)
clf.fit(X, Y)
joblib.dump(clf, 'Model(SGDC).pkl')

# Test predictions on example reviews
pos_review = 'I have bought several of the Vitality canned dog food products and have found them all to be of good quality.\
The product looks more like a stew than a processed meat and it smells better. My Labrador is finicky and she appreciates this product better than  most.'
neg_review = 'Product arrived labeled as Jumbo Salted Peanuts...the peanuts were actually small sized unsalted. \
Not sure if this was an error or if the vendor intended to represent the product as "Jumbo".'
neutral_review = 'At the end of the day, a dumbbell is just that, but the finishing of the 40lb dumbbell purchased was not pristine. \
Quality control appears to be joke. Flicks of the coating were still hanging and several corners were blunted. \
I got a replacement but it seems to be a slight improvement. If you are not a stickler for things like this, then by all means go for it.'

if predict(pos_review) == "Positive" and predict(neg_review) == "Negative" and predict(neutral_review) == "Neutral":
    print("Model works fine, ready for deployment")
