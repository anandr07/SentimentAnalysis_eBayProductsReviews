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
# from data_preprocessing import clean_text, preprocess_text, sentence_to_words

#%%
# Load your eBay reviews dataset
# Replace 'your_ebay_dataset_path.csv' with the actual path to your eBay dataset
df= pd.read_csv('ebay_reviews.csv')

# Display the head of the eBay dataset
print(df.head())

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

# %%
# Here I have implemented KNN algorithm using sklearn using BOW, tf-idf, Average word2vec and tf-idf word2vec
import warnings
warnings.filterwarnings("ignore")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay

#%%
#find knn to simple cross validation with Brute Force and KD-Tree
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

def KNN_train_simple_cv(X_train, Y_train, X_test, Y_test):
    X_tr, X_cv, Y_tr, Y_cv = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

    k = []
    pred_cv_auc = []
    pred_train_auc = []
    pred_cv_accuracy = []
    pred_train_accuracy = []
    max_roc_auc = -1
    best_k_auc = 0
    best_accuracy = 0
    for i in range(1, 24, 2):
        knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute', n_jobs=-1)
        knn.fit(X_tr, Y_tr)
        probs_cv = knn.predict_proba(X_cv)
        probs_train = knn.predict_proba(X_tr)

        auc_score_cv = roc_auc_score(Y_cv, probs_cv, multi_class='ovr')  # find AUC score
        auc_score_train = roc_auc_score(Y_tr, probs_train, multi_class='ovr')

        # Calculate accuracy
        accuracy_cv = accuracy_score(Y_cv, np.argmax(probs_cv, axis=1))
        accuracy_train = accuracy_score(Y_tr, np.argmax(probs_train, axis=1))

        print(f"{i} - AUC Score (CV): {auc_score_cv}  Accuracy (CV): {accuracy_cv}")
        pred_cv_auc.append(auc_score_cv)
        pred_train_auc.append(auc_score_train)
        pred_cv_accuracy.append(accuracy_cv)
        pred_train_accuracy.append(accuracy_train)
        k.append(i)

        if max_roc_auc < auc_score_cv:
            max_roc_auc = auc_score_cv
            best_k_auc = i
        if best_accuracy < accuracy_cv:
            best_accuracy = accuracy_cv

    print(f"Best k-value based on AUC: {best_k_auc}")
    print(f"Best accuracy: {best_accuracy}")

    # Plotting k vs AUC Score
    plt.plot(k, pred_cv_auc, 'r-', label='CV AUC Score')
    plt.plot(k, pred_train_auc, 'g-', label='Train AUC Score')
    plt.legend(loc='upper right')
    plt.title("k vs AUC Score")
    plt.ylabel('AUC Score')
    plt.xlabel('k')
    plt.show()

    # Plotting k vs Accuracy
    plt.plot(k, pred_cv_accuracy, 'b-', label='CV Accuracy')
    plt.plot(k, pred_train_accuracy, 'c-', label='Train Accuracy')
    plt.legend(loc='upper right')
    plt.title("k vs Accuracy")
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.show()

    # Confusion Matrix for best k based on accuracy
    knn = KNeighborsClassifier(n_neighbors=best_k_auc, algorithm='brute')
    knn.fit(X_train, Y_train)
    probs_test = knn.predict_proba(X_test)
    cm = confusion_matrix(Y_test, np.argmax(probs_test, axis=1))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix Test')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Calculate and print AUC score
    auc_score = roc_auc_score(Y_test, probs_test, multi_class='ovr')
    print(f"AUC Score (Test): {auc_score}")

    # Calculate and print accuracy
    accuracy = accuracy_score(Y_test, np.argmax(probs_test, axis=1))
    print(f"Accuracy (Test): {accuracy}")

    return auc_score, accuracy


# def KNN_train_simple_cv(X_train, Y_train, X_test, Y_test):
#     X_tr, X_cv, Y_tr, Y_cv = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

#     k = []
#     pred_cv_auc = []
#     pred_train_auc = []
#     pred_cv_accuracy = []
#     pred_train_accuracy = []
#     max_roc_auc = -1
#     best_k_auc = 0
#     best_accuracy = 0
#     for i in range(1, 24, 2):
#         knn = KNeighborsClassifier(n_neighbors=i, algorithm='brute',n_jobs=-1)
#         knn.fit(X_tr, Y_tr)
#         probs_cv = knn.predict_proba(X_cv)[:, 1]
#         probs_train = knn.predict_proba(X_tr)[:, 1]

#         auc_score_cv = roc_auc_score(Y_cv, probs_cv, multi_class='ovr')  # find AUC score
#         auc_score_train = roc_auc_score(Y_tr, probs_train, multi_class='ovr')

#         # Calculate accuracy
#         threshold = 0.5
#         binary_preds_cv = (probs_cv > threshold).astype(int)
#         binary_preds_train = (probs_train > threshold).astype(int)
#         accuracy_cv = accuracy_score(Y_cv, binary_preds_cv)
#         accuracy_train = accuracy_score(Y_tr, binary_preds_train)

#         print(f"{i} - AUC Score (CV): {auc_score_cv}  Accuracy (CV): {accuracy_cv}")
#         pred_cv_auc.append(auc_score_cv)
#         pred_train_auc.append(auc_score_train)
#         pred_cv_accuracy.append(accuracy_cv)
#         pred_train_accuracy.append(accuracy_train)
#         k.append(i)

#         if max_roc_auc < auc_score_cv:
#             max_roc_auc = auc_score_cv
#             best_k_auc = i
#         if best_accuracy < accuracy_cv:
#             best_accuracy = accuracy_cv

#     print(f"Best k-value based on AUC: {best_k_auc}")
#     print(f"Best accuracy: {best_accuracy}")

#     # Plotting k vs AUC Score
#     plt.plot(k, pred_cv_auc, 'r-', label='CV AUC Score')
#     plt.plot(k, pred_train_auc, 'g-', label='Train AUC Score')
#     plt.legend(loc='upper right')
#     plt.title("k vs AUC Score")
#     plt.ylabel('AUC Score')
#     plt.xlabel('k')
#     plt.show()

#     # Plotting k vs Accuracy
#     plt.plot(k, pred_cv_accuracy, 'b-', label='CV Accuracy')
#     plt.plot(k, pred_train_accuracy, 'c-', label='Train Accuracy')
#     plt.legend(loc='upper right')
#     plt.title("k vs Accuracy")
#     plt.ylabel('Accuracy')
#     plt.xlabel('k')
#     plt.show()

#     # Confusion Matrix for best k based on accuracy
#     knn = KNeighborsClassifier(n_neighbors=best_k_auc, algorithm='brute')
#     knn.fit(X_train, Y_train)
#     prob_cv = knn.predict_proba(X_cv)[:, 1]
#     binary_preds_cv = (prob_cv > threshold).astype(int)
#     cm = confusion_matrix(Y_cv, binary_preds_cv)

#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
#     plt.title('Confusion Matrix Train')
#     plt.colorbar()
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')

#     plt.show()

#     prob_test = knn.predict_proba(X_test)[:, 1]
#     binary_preds_test = (prob_test > threshold).astype(int)
#     cm = confusion_matrix(Y_test, binary_preds_test)

#     # Use ConfusionMatrixDisplay for visualization
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
#     disp.plot()
#     plt.title('Confusion Matrix Test')
#     plt.show()  

#     print(Y_test,"\n")
#     print(prob_test,"\n")
#     print(binary_preds_test,"\n")

#     # Calculate and print AUC score
#     auc_score = roc_auc_score(Y_test, prob_test)
#     print(f"AUC Score (Test): {auc_score}")

#     # Calculate and print accuracy
#     accuracy = accuracy_score(Y_test, binary_preds_test)
#     print(f"Accuracy (Test): {accuracy}")

#     return auc_score, accuracy


#%%
#NAIVE BAYE's
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import sparse

def NaiveBayes_train_simple_cv(X_train, Y_train, X_test, Y_test):
    # Split the training data for cross-validation
    X_tr, X_cv, Y_tr, Y_cv = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

    # Initialize the classifier
    nb = MultinomialNB()

    # Train the classifier
    nb.fit(X_tr, Y_tr)

    # Predict probabilities for CV and training sets
    probs_cv = nb.predict_proba(X_cv)
    probs_train = nb.predict_proba(X_tr)

    # Calculate AUC score for CV and training sets
    auc_score_cv = roc_auc_score(Y_cv, probs_cv, multi_class='ovr')
    auc_score_train = roc_auc_score(Y_tr, probs_train, multi_class='ovr')

    # Calculate accuracy for CV and training sets
    accuracy_cv = accuracy_score(Y_cv, nb.predict(X_cv))
    accuracy_train = accuracy_score(Y_tr, nb.predict(X_tr))

    print(f"AUC Score (CV): {auc_score_cv}  Accuracy (CV): {accuracy_cv}")
    print(f"AUC Score (Train): {auc_score_train}  Accuracy (Train): {accuracy_train}")

    # Confusion Matrix for CV set
    cm = confusion_matrix(Y_cv, nb.predict(X_cv))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix Train')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Confusion Matrix for test set
    prob_test = nb.predict_proba(X_test)
    cm = confusion_matrix(Y_test, nb.predict(X_test))

    # Use ConfusionMatrixDisplay for visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(Y_test))
    disp.plot()
    plt.title('Confusion Matrix Test')
    plt.show()  

    # Calculate and print AUC score for test set
    auc_score = roc_auc_score(Y_test, prob_test, multi_class='ovr')
    print(f"AUC Score (Test): {auc_score}")

    # Calculate and print accuracy for test set
    accuracy = accuracy_score(Y_test, nb.predict(X_test))
    print(f"Accuracy (Test): {accuracy}")

    return auc_score, accuracy


# # Train Multinomial Naive Bayes model and get AUC score and accuracy
# def NaiveBayes_train_simple_cv(X_train, Y_train, X_test, Y_test):
#     # Split the training data for cross-validation
#     X_tr, X_cv, Y_tr, Y_cv = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

#     # Initialize the classifier
#     nb = MultinomialNB()

#     # Train the classifier
#     nb.fit(X_tr, Y_tr)

#     # Predict probabilities for CV and training sets
#     probs_cv = nb.predict_proba(X_cv)[:, 1]
#     probs_train = nb.predict_proba(X_tr)[:, 1]

#     # Calculate AUC score for CV and training sets
#     auc_score_cv = roc_auc_score(Y_cv, probs_cv)
#     auc_score_train = roc_auc_score(Y_tr, probs_train)

#     # Calculate accuracy for CV and training sets
#     threshold = 0.5
#     binary_preds_cv = (probs_cv > threshold).astype(int)
#     binary_preds_train = (probs_train > threshold).astype(int)
#     accuracy_cv = accuracy_score(Y_cv, binary_preds_cv)
#     accuracy_train = accuracy_score(Y_tr, binary_preds_train)

#     print(f"AUC Score (CV): {auc_score_cv}  Accuracy (CV): {accuracy_cv}")
#     print(f"AUC Score (Train): {auc_score_train}  Accuracy (Train): {accuracy_train}")

#     # Confusion Matrix for CV set
#     cm = confusion_matrix(Y_cv, binary_preds_cv)
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
#     plt.title('Confusion Matrix Train')
#     plt.colorbar()
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.show()

#     # Confusion Matrix for test set
#     prob_test = nb.predict_proba(X_test)[:, 1]
#     binary_preds_test = (prob_test > threshold).astype(int)
#     cm = confusion_matrix(Y_test, binary_preds_test)

#     # Use ConfusionMatrixDisplay for visualization
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
#     disp.plot()
#     plt.title('Confusion Matrix Test')
#     plt.show()  

#     # Calculate and print AUC score for test set
#     auc_score = roc_auc_score(Y_test, prob_test)
#     print(f"AUC Score (Test): {auc_score}")

#     # Calculate and print accuracy for test set
#     accuracy = accuracy_score(Y_test, binary_preds_test)
#     print(f"Accuracy (Test): {accuracy}")

#     return auc_score, accuracy

#%%
#SGD Classifier
#%%
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from sklearn.metrics import make_scorer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Assuming X_train_tfidf, Y_train, X_test_tfidf, Y_test are loaded

# Define the parameter grid for Random Search CV
param_dist = {
    'alpha': loguniform(1e-6, 1e-1),
    'eta0': [0.01, 0.1, 0.2, 0.5],
}

# Define custom scorer for roc_auc_score
roc_auc_scorer = make_scorer(roc_auc_score, greater_is_better=True, needs_proba=True)

# Train SGDClassifier model and perform Random Search CV
# Train SGDClassifier model and perform Random Search CV
def SGDClassifier_train_random_search_cv(X_train, Y_train, X_test, Y_test):
    # Split the training data for cross-validation
    X_tr, X_cv, Y_tr, Y_cv = train_test_split(X_train, Y_train, test_size=0.33, random_state=0)

    # Initialize the classifier
    sgd_classifier = SGDClassifier(loss='log_loss', random_state=0)

    # Define RandomizedSearchCV
    random_search = RandomizedSearchCV(sgd_classifier, param_distributions=param_dist, n_iter=10, scoring=roc_auc_scorer, cv=3, random_state=0)

    # Perform RandomizedSearchCV
    random_search.fit(X_tr, Y_tr)

    # Get the best hyperparameters
    best_params = random_search.best_params_
    print(f"Best Hyperparameters: {best_params}")

    # Use the best hyperparameters to train the final model
    best_sgd_classifier = random_search.best_estimator_
    best_sgd_classifier.fit(X_tr, Y_tr)

    # Predict probabilities for CV and training sets
    probs_cv = best_sgd_classifier.predict_proba(X_cv)[:, :]
    probs_train = best_sgd_classifier.predict_proba(X_tr)[:, :]
    print(Y_cv.shape)
    print(probs_cv.shape)
    # Calculate AUC score for CV and training sets
    auc_score_cv = roc_auc_score(Y_cv, probs_cv, multi_class='ovr')
    auc_score_train = roc_auc_score(Y_tr, probs_train, multi_class='ovr')

    # Calculate accuracy for CV and training sets
    threshold = 0.5
    binary_preds_cv = (probs_cv > threshold).astype(int)
    binary_preds_train = (probs_train > threshold).astype(int)
    accuracy_cv = accuracy_score(Y_cv, best_sgd_classifier.predict(X_cv))
    accuracy_train = accuracy_score(Y_tr, best_sgd_classifier.predict(X_tr))

    print(f"AUC Score (CV): {auc_score_cv}  Accuracy (CV): {accuracy_cv}")
    print(f"AUC Score (Train): {auc_score_train}  Accuracy (Train): {accuracy_train}")

    # Confusion Matrix for CV set
    cm = confusion_matrix(Y_cv, best_sgd_classifier.predict(X_cv))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Confusion Matrix Train')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Confusion Matrix for test set
    prob_test = best_sgd_classifier.predict_proba(X_test)[:, :]
    binary_preds_test = (prob_test > threshold).astype(int)
    predicted_labels_test = best_sgd_classifier.predict(X_test)

    cm = confusion_matrix(Y_test, predicted_labels_test)

    # Use ConfusionMatrixDisplay for visualization
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(Y_test))
    disp.plot()
    plt.title('Confusion Matrix Test')
    plt.show()

    print(Y_test, "\n")
    print(prob_test, "\n")
    print(binary_preds_test, "\n")

    # Calculate and print AUC score for test set
    auc_score = roc_auc_score(Y_test, prob_test,multi_class='ovr')
    print(f"AUC Score (Test): {auc_score}")

    # Calculate and print accuracy for test set
    accuracy = accuracy_score(Y_test, predicted_labels_test)
    print(f"Accuracy (Test): {accuracy}")

    return auc_score, accuracy

# Call the function with your data
# SGDClassifier_train_random_search_cv(X_train_bow, Y_train, X_test_bow, Y_test)


#%%
# Converting sentence into words using sentence_to_words function from the data_preprocessing file
list_of_words_in_sentence = sentence_to_words(df, 'Clean_Text')
print(list_of_words_in_sentence)
#%%
# # Visualizations
# # Assuming you have a 'Clean_Text' column after preprocessing
# positive_text = " ".join(df[df['Review'] == "Positive"]['Clean_Text'])
# negative_text = " ".join(df[df['Review'] == "Negative"]['Clean_Text'])

# # Generate WordCloud for positive reviews
# positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)

# # Generate WordCloud for negative reviews
# negative_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(negative_text)

# # Plot the WordClouds
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.imshow(positive_wordcloud, interpolation='bilinear')
# plt.title('Word Cloud for Positive Reviews')
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.imshow(negative_wordcloud, interpolation='bilinear')
# plt.title('Word Cloud for Negative Reviews')
# plt.axis("off")
# plt.show()

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

auc_score_bow_test_SGDC, accuracy_bow_test_SGDC = SGDClassifier_train_random_search_cv(X_train_bow_ebay, Y_train, X_test_bow_ebay, Y_test)

#%%
# SGD Classifier on tf-idf

auc_score_tf_idf_test_SGDC, accuracy_tf_idf_test_SGDC = SGDClassifier_train_random_search_cv(X_train_tfidf, Y_train, X_test_tfidf, Y_test)

#%%
# SGD Classifier on Tf-Idf word2vec 

auc_score_tfidf_word2vec_test_SGDC, accuracy_tfidf_word2vec_test_SGDC = SGDClassifier_train_random_search_cv(X_train_tfidf_word2vec, Y_train_tfidf_wor2vec, X_test_tfidf_word2vec, Y_test_tfidf_wor2vec)

# %%
