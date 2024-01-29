#%%[markdown]
## Sentiment prediction from Ebay reviews

### About DataSet

# This dataset consists of reviews of product reviews from Ebay.

### Contents

# - **ebay_reviews.csv :** Contains tabular data of Category, Review Title, Review Content and Rating.

### Data includes:
 
# - 44,756 reviews 

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, normalize
from tqdm import tqdm
import numpy as np
# from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import matplotlib.pyplot as plt
# from wordcloud import WordCloud

#%%
# Load your eBay reviews dataset
# Replace 'your_ebay_dataset_path.csv' with the actual path to your eBay dataset
df = pd.read_csv(r"C:\Anand\Projects_GWU\Semtiment_Analysis_ebay_product_reviews\ebay_reviews.csv\ebay_reviews.csv")

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

# Giving 4&5 as Positive and 1&2 as Negative Rating
def assign_values(value):
    if value < 3:
        return 'Negative'
    else:
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