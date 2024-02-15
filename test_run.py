import joblib

# Load the CountVectorizer and trained model
count_vect = joblib.load('CountVectorizer.pkl')
clf = joblib.load('Model(SGDC).pkl')

def clean_text(sentence):
    # Your clean_text function code here...

def predict_review_sentiment(review):
    # Preprocess the input review
    cleaned_review = clean_text(review)
    
    # Transform the review using the CountVectorizer
    review_vect = count_vect.transform([cleaned_review])
    
    # Predict the sentiment using the trained model
    prediction = clf.predict(review_vect)[0]
    
    if prediction == 0:
        return "Negative"
    elif prediction == 1:
        return "Neutral"
    else:
        return "Positive"

# Test your model by providing a review
review = "This product is excellent! I highly recommend it."
sentiment = predict_review_sentiment(review)
print("Predicted sentiment:", sentiment)
