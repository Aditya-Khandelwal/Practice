#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import warnings

df = pd.read_csv("sentimentdataset.csv", encoding='ISO-8859-1')

print(df.head())

print(df.isnull().sum())

df.dropna(subset=['New_Date', 'clean_tweet'], inplace=True)

df['New_Date'] = pd.to_datetime(df['New_Date'], dayfirst=True)

df.set_index('New_Date', inplace=True)


# In[4]:


nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()

# Calculate sentiment scores for each tweet
df['sentiment_scores'] = df['clean_tweet'].apply(lambda tweet: sid.polarity_scores(tweet))

# Extract compound score
df['compound_score'] = df['sentiment_scores'].apply(lambda score_dict: score_dict['compound'])

# Categorize sentiments based on compound score
df['sentiment'] = df['compound_score'].apply(
    lambda score: 'positive' if score >= 0.05 else ('negative' if score <= -0.05 else 'neutral')
)

print(df[['clean_tweet', 'compound_score', 'sentiment']].head())


# In[5]:


# Resample data to daily sentiment counts
dfd = pd.DataFrame(index=pd.date_range(start=df.index.min(), end=df.index.max()))
dfd['positive'] = df[df['sentiment'] == 'positive'].resample('D').size()
dfd['negative'] = df[df['sentiment'] == 'negative'].resample('D').size()
dfd['neutral'] = df[df['sentiment'] == 'neutral'].resample('D').size()

# Fill NaN values with 0 to represent days with no sentiments
dfd.fillna(0, inplace=True)

plt.figure(figsize=(12, 6))
plt.plot(dfd.index, dfd['positive'], label='Positive', color='g')
plt.plot(dfd.index, dfd['negative'], label='Negative', color='r')
plt.plot(dfd.index, dfd['neutral'], label='Neutral', color='b')
plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.ylabel('Count')
plt.legend()
plt.show()


# In[8]:


# Summarize data
sentiment_summary = dfd.describe()

plt.figure(figsize=(12, 6))
plt.plot(dfd.index, dfd['positive'], label='Positive', color='g')
plt.plot(dfd.index, dfd['negative'], label='Negative', color='r')
plt.plot(dfd.index, dfd['neutral'], label='Neutral', color='b')
plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.savefig('sentiment_trends.png')
plt.show()

# Export summarized data to CSV
sentiment_summary.to_csv('sentiment_summary.csv')

print("Report generated: 'sentiment_summary.csv' and 'sentiment_trends.png'")


# In[ ]:




