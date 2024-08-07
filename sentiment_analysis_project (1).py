#!/usr/bin/env python
# coding: utf-8

# ****Sentiment Analysis on Cab Service Company Reviews Using suitable Naive Bayes algorithms.****

# In[70]:


import numpy as np
from sklearn.naive_bayes import  BernoulliNB,MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords
import string
import nltk


# In[71]:


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[72]:


import pandas as pd
df = pd.read_csv('dataset.csv')


# In[73]:


df


# **Performing Data cleaning**

# In[74]:


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


#  **Removing emojis from the dataset for better predictions**

# In[75]:


import re

# Function to remove emojis from a given text
def remove_emojis(text):
    if isinstance(text, str):  # Check if the input is a string
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251" 
            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    else:
        return text

# Read the CSV file
input_file = 'dataset.csv'

# Apply the remove_emojis function to all text columns
for column in df.select_dtypes(include=['object']).columns:
    df[column] = df[column].apply(remove_emojis)

print(f'Emojis removed and cleaned data saved ')


# In[76]:


df['Text'] = df['Text'].apply(preprocess_text)
df1=df.copy()
df1.head(50)


# In[77]:


x=df1['Text'].str.lower()
y=df1['Sentiment'].str.lower()


# In[78]:


x


# In[79]:


y


# In[80]:


df['Sentiment'].value_counts()


# In[81]:


from sklearn.feature_extraction.text import CountVectorizer


# In[82]:


vectorizer1=CountVectorizer(binary=True)
vectorizer2=CountVectorizer(binary=False)


# In[83]:


x1=vectorizer1.fit_transform(x)
x2=vectorizer2.fit_transform(x)


# In[84]:


from sklearn.model_selection import train_test_split


# In[85]:


xtrain1,xtest1,ytrain,ytest=train_test_split(x1,y,test_size=0.25,random_state=42)


# In[86]:


xtrain2,xtest2,ytrain,ytest=train_test_split(x2,y,test_size=0.25,random_state=42)


# **Using algorithms such as Bernoulli Naive Bayes and Multinomial Naive Bayes, and techniques such as Count Vectorization and TfidfVectorizer, we will determine which approach is the most accurate for this dataset.**

# In[87]:


bnb=BernoulliNB()
mnb=MultinomialNB()


# In[88]:


bnb.fit(xtrain1,ytrain)


# In[89]:


mnb.fit(xtrain2,ytrain)


# In[90]:


y_pred1=bnb.predict(xtest1)


# In[91]:


y_pred2=mnb.predict(xtest2)


# In[92]:


from sklearn.metrics import accuracy_score


# In[93]:


accuracy_score(ytest,y_pred1)


# In[94]:


accuracy_score(ytest,y_pred2)


# In[95]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[96]:


xtrain3,xtest3,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=42)


# In[97]:


from sklearn.pipeline import make_pipeline
model=make_pipeline(TfidfVectorizer(),MultinomialNB())


# In[98]:


model.fit(xtrain3,ytrain)


# In[99]:


predictions_tf=model.predict(xtest3)


# In[100]:


accuracy_score(ytest,predictions_tf)


# **Conclusion: The Bernoulli classification algorithm achieved the best accuracy score of 0.7413909520594193 using the Count Vectorization technique.**

# **Now by taking dynamic input from the user, we can verify if the model is providing accurate predictions.**

# In[101]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

# Assuming x, xtrain1, xtest1, ytrain, ytest are already defined

# Create and fit the CountVectorizer
vectorizer1 = CountVectorizer(binary=True)
x1 = vectorizer1.fit_transform(x)

# Train the Bernoulli Naive Bayes model
bnb = BernoulliNB()
bnb.fit(xtrain1, ytrain)

# Predict on test data and calculate accuracy
y_pred1 = bnb.predict(xtest1)
print("Accuracy:", accuracy_score(ytest, y_pred1))

# Function to preprocess the review
def preprocess_text(text):
    # Add your preprocessing steps here (e.g., lowercasing, removing punctuation)
    return text.lower()

def predict_rating(review):
    # Preprocess the review
    preprocessed_review = preprocess_text(review)

    # Transform the preprocessed review using the fitted CountVectorizer
    review_vectorized = vectorizer1.transform([preprocessed_review])

    # Predict the rating using the trained Bernoulli Naive Bayes model
    predicted_rating = bnb.predict(review_vectorized)[0]

    return predicted_rating

# Get user input for the review
user_review = input("Enter your review: ")

# Predict the rating
predicted_rating = predict_rating(user_review)

print("Predicted Rating:", predicted_rating)


# **Saving the Bernoulli model, which has demonstrated the highest accuracy, to disk for easy access using Joblib library.**

# In[102]:


import joblib


# In[103]:


model = 'bernouli.joblib'
joblib.dump(bnb, model)

