#%%[markdown]
## Sentiment prediction from Ebay reviews

### About DataSet

# This dataset consists of reviews of product reviews from Ebay.

### Contents

# - **ebay_reviews.csv :** Contains tabular data of Category, Review Title, Review Content and Rating.

### Data includes:
 
# - 44,756 reviews 

#%%
import os, sys
import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from ML_algo_implemented.KNN import KNN_train_simple_cv
from ML_algo_implemented.NaiveBayes import NaiveBayes_train_simple_cv
from ML_algo_implemented.SGDClassifier import SGDClassifier_train_random_search_cv
# from data_preprocessing import clean_text, preprocess_text, sentence_to_words

#%%
# Load your eBay reviews dataset
# Replace 'your_ebay_dataset_path.csv' with the actual path to your eBay dataset
df= pd.read_csv('ebay_reviews.csv')

# Display the head of the eBay dataset
print(df.head())

# df = df[:1000]
#%%
# Number of unique categories
num_categories = df['category'].nunique()
print(f"Number of unique categories in eBay dataset: {num_categories}")

# Unique rating scores
unique_ratings = df['rating'].unique()
print(f"Unique rating scores in eBay dataset: {unique_ratings}")

# Drop unnecessary columns
df = df[['category', 'review title', 'review content', 'rating']]

#%%
# Combine review title and review content into a single 'Clean_Text' column
df['Clean_Text'] = df['review title'] + ' ' + df['review content']

# Giving 4&5 as Positive and 1&2 as Negative Rating, 3 as Neutral
def assign_values(value):
    if value < 3:
        return 'Negative'
    elif value == 3:
        return 'Neutral'
    else :
        return 'Positive'

df['Review'] = df['rating'].apply(assign_values)

#%%
# Check for duplicate reviews
boolean = df['Clean_Text'].duplicated().any()
print(f"Duplicate reviews present: {boolean}")

# Drop duplicated reviews
df = df.drop_duplicates(subset='Clean_Text', keep='first')

#%%
# Function to clean text
def clean_text(text):
    ''' This function removes punctuations, HTML tags, URLs, and Non-Alpha Numeric words.
    '''
    unwanted_chars_patterns = [
        r'[!?,;:—".]',  # Remove punctuation 
        r'<[^>]+>',  # Remove HTML tags
        r'http[s]?://\S+',  # Remove URLs
        r"^[A-Za-z]+$" # Non-Alpha Numeric
    ]

    for pattern in unwanted_chars_patterns:
        text = re.sub(pattern, '', str(text))

    return text

#%%
# Apply text preprocessing to the 'Clean_Text' column
df['Clean_Text'] = df['Clean_Text'].apply(lambda x: clean_text(x))


#%%
#Other definitions defined here

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
nltk.download('punkt')

def preprocess_text(text):
    ''' This function performs tokenization of text and also uses Snowball Stemmer for stemming of words.
    '''
    # Tokenizing the text and removing stopwords
    tokens = nltk.word_tokenize(text)
    # tokens = [word for word in tokens if word not in stop_words]
    tokens = [word for word in tokens if word not in stop_words and word.isalpha() and len(word) >= 3]
    # Applying Snowball stemming
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def sentence_to_words(data_frame, column_name):
    ''' This function converts a sentance in words keeping words that are alpha-numeric only.
        Also makes all the words to lowercase
    '''
    i = 0
    list_of_words_in_sentance = []

    for sent in data_frame[column_name].values:
        list_of_words_in_filtered_sentence = []
        sent = clean_text(sent)
        
        # Split the sentence into words
        words = sent.split()

        # Check if each word is alphanumeric (Just for a double check)
        for word in words:
            if word.isalnum():
                list_of_words_in_filtered_sentence.append(word.lower())
        
        list_of_words_in_sentance.append(list_of_words_in_filtered_sentence)

    return list_of_words_in_sentance

#%%
#%%[markdown]
# Label Encoding Reviews Column
label_encoder = LabelEncoder()

# Fit and transform the "Review" column
df['Review'] = label_encoder.fit_transform(df['Review'])


#%%
# Converting sentence into words using sentence_to_words function from the data_preprocessing file
list_of_words_in_sentence = sentence_to_words(df, 'Clean_Text')
print(list_of_words_in_sentence)
#%%
# Visualizations
# Assuming you have a 'Clean_Text' column after preprocessing
positive_text = " ".join(df[df['Review'] == 2]['Clean_Text'])
negative_text = " ".join(df[df['Review'] == 0]['Clean_Text'])
neutral_text = " ".join(df[df['Review'] == 1]['Clean_Text'])

# Generate WordCloud for positive reviews
positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

# Generate WordCloud for negative reviews
negative_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

# Generate WordCloud for Neutral reviews
neutral_wordcloud = WordCloud(width=800, height=400, background_color='pink', colormap='Reds').generate(neutral_text)

# Plot the WordClouds
plt.figure(figsize=(18, 6))  # Increase the width to accommodate the third subplot
plt.subplot(1, 3, 1)  # Modify the subplot parameters to add a third subplot
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Reviews')
plt.axis("off")

plt.subplot(1, 3, 2)  # Add a third subplot for the neutral WordCloud
plt.imshow(neutral_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Neutral Reviews')
plt.axis("off")

plt.subplot(1, 3, 3)  # Add a third subplot for the negative WordCloud
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Reviews')
plt.axis("off")

plt.show()

#%%
# Train and Test Split
X = df['Clean_Text']
Y = df['Review']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=123)

#%%
# Bag Of Words for eBay dataset
Count_vectorizer_ebay = CountVectorizer()
X_train_bow_ebay = Count_vectorizer_ebay.fit_transform(X_train.values)
X_test_bow_ebay = Count_vectorizer_ebay.transform(X_test.values)

# Normalize BOW Train and Test Data for eBay dataset
X_train_bow_ebay = normalize(X_train_bow_ebay)
X_test_bow_ebay = normalize(X_test_bow_ebay)

# Display the shape of your processed eBay data
print("Shape of eBay data:", X_train_bow_ebay.shape, X_test_bow_ebay.shape, Y_train.shape, Y_test.shape)

#%%
Count_vectorizer_n_grams = CountVectorizer(ngram_range=(1,3) ) 
X_train_n_grams = Count_vectorizer_n_grams.fit_transform(X_train.values)
print("Shape of dataset after converting into uni, bi and tri-grams is ",X_train_n_grams.get_shape())
X_test_n_grams = Count_vectorizer_n_grams.transform(X_test.values)
print("Shape of dataset after converting into uni, bi and tri-grams is ",X_test_n_grams.get_shape())

# %%
# tf-idf Vectorizer
tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X_train_tf_idf_vectorizer = tf_idf_vectorizer.fit_transform(X_train.values)
X_test_tf_idf_vectorizer = tf_idf_vectorizer.transform(X_test.values)

print("Shape of train dataset after converting into tf-idf is ", X_train_tf_idf_vectorizer.get_shape())
print("Shape of test dataset after converting into tf-idf is ", X_test_tf_idf_vectorizer.get_shape())

# %%
# Normalize Tf-Idf Train and Test Data
X_train_tfidf = normalize(X_train_tf_idf_vectorizer)
X_test_tfidf = normalize(X_test_tf_idf_vectorizer)
print("Train Data Size: ",X_train_tfidf.get_shape())
print("Test Data Size: ",X_test_tfidf.shape)
# %%
## word2vec Model
# Making word2vec model using our data set and the same model will be used further.

# Training word2vec model on our own data.
w2v_model=gensim.models.Word2Vec(list_of_words_in_sentence,min_count=5, workers=4) 

# %%
# Saving the vocabolary of words in our trained word2vec model
w2v_vocab = list(w2v_model.wv.key_to_index)
# %%
# Get the top 10 words most similar words to "quality"
w2v_model.wv.most_similar('good')

# %%
sent_vectors_avg_word2vec = []; # The avg-w2v for each sentence/review is stored in this list
vector_size = len(w2v_model.wv['good']) 

for sent in tqdm(list_of_words_in_sentence): # Iterating over each review/sentence
    sent_vec = np.zeros(vector_size) 
    cnt_words =0; 
    for word in sent: # Iterating over each word in a review/sentence
        if word in w2v_vocab:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
    if cnt_words != 0:
        sent_vec /= cnt_words
    sent_vectors_avg_word2vec.append(sent_vec)
print(len(sent_vectors_avg_word2vec))

# %%
X_train_avg_wor2vec, X_test_avg_wor2vec, Y_train_avg_wor2vec, Y_test_avg_wor2vec = train_test_split(sent_vectors_avg_word2vec,Y, test_size=.20, random_state=0)
X_train_avg_wor2vec=normalize(X_train_avg_wor2vec)
X_test_avg_wor2vec=normalize(X_test_avg_wor2vec)
print(X_train_avg_wor2vec.shape)
print(X_test_avg_wor2vec.shape)

# %%
## Tf-Idf Word2vec 

tfidf_model = TfidfVectorizer()
tf_idf_matrix = tfidf_model.fit_transform(df['Clean_Text'].values)
# we are converting a dictionary with word as a key, and the idf as a value
dictionary = dict(zip(tfidf_model.get_feature_names_out(), list(tfidf_model.idf_)))

# TF-IDF weighted Word2Vec
tfidf_feat = tfidf_model.get_feature_names_out() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0
for sent in tqdm(list_of_words_in_sentence): # for each review/sentence 
    sent_vec = np.zeros(vector_size) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        if word in w2v_vocab:
            vec = w2v_model.wv[word]
            if word in dictionary:
                tf_idf = dictionary[word]*(sent.count(word)/len(sent))
                sent_vec += (vec * tf_idf)
                weight_sum += tf_idf
            else:
                pass 
       
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1

X_train_tfidf_word2vec, X_test_tfidf_word2vec, Y_train_tfidf_wor2vec, Y_test_tfidf_wor2vec = train_test_split(tfidf_sent_vectors, Y, test_size=0.20,random_state=0)
X_train_tfidf_word2vec=normalize(X_train_tfidf_word2vec)
X_test_tfidf_word2vec=normalize(X_test_tfidf_word2vec)
print(X_train_tfidf_word2vec.shape)
print(X_test_tfidf_word2vec.shape)

# %%
# KNN on Bag of Words

auc_score_bow_test_KNN, accuracy_bow_test_KNN = KNN_train_simple_cv(X_train_bow_ebay, Y_train, X_test_bow_ebay, Y_test)

# %%
# KNN on tf-idf

auc_score_tf_idf_test_KNN, accuracy_tf_idf_test_KNN = KNN_train_simple_cv(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

#%%
# KNN on Tf-Idf word2vec 

auc_score_word2vec_test_KNN, accuracy_bword2vec_test_KNN = KNN_train_simple_cv(X_train_tfidf_word2vec, Y_train_tfidf_wor2vec, X_test_tfidf_word2vec, Y_test_tfidf_wor2vec)

#%%
# Naive Bayes from here
# NaiveBayes on Bag of Words

auc_score_bow_test_NB, accuracy_bow_test_NB = NaiveBayes_train_simple_cv(X_train_bow_ebay, Y_train, X_test_bow_ebay, Y_test)

# %%
# NaiveBayes on tf-idf

auc_score_tf_idf_test_NB, accuracy_tf_idf_test_NB = NaiveBayes_train_simple_cv(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

# %%
# SGD Classifier on Bag of Words

auc_score_bow_test_SGDC, accuracy_bow_test_SGDC, best_sgd_classifier_bow = SGDClassifier_train_random_search_cv(X_train_bow_ebay, Y_train, X_test_bow_ebay, Y_test)

#%%
# SGD Classifier on tf-idf

auc_score_tf_idf_test_SGDC, accuracy_tf_idf_test_SGDC, best_sgd_classifier_tf_idf = SGDClassifier_train_random_search_cv(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

#%%
# SGD Classifier on Tf-Idf word2vec 

auc_score_tfidf_word2vec_test_SGDC, accuracy_tfidf_word2vec_test_SGDC, best_sgd_classifier_tfidf_word2vec = SGDClassifier_train_random_search_cv(X_train_tfidf_word2vec, Y_train_tfidf_wor2vec, X_test_tfidf_word2vec, Y_test_tfidf_wor2vec)

# %%
import joblib

# Assuming you have trained the SGD Classifier model and stored it in a variable named clf

# Save the trained SGD Classifier model to a file named "SGDClassifier_model.pkl"
joblib.dump(best_sgd_classifier_tfidf_word2vec, 'SGDClassifier_model.pkl')

print("SGD Classifier model saved as SGDClassifier_model.pkl")

# %%
