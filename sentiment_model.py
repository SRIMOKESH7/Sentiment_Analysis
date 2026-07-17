#import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Predicting the sentiment of a given sentence using VADER
def predict_sentiment_vader(sentence):
    sentiment = analyzer.polarity_scores(sentence)['compound']
    if sentiment >= 0.05:
        return 'positive'
    elif sentiment <= -0.05:
        return 'negative'
    else:
        return 'neutral'

