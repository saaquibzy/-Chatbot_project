# from textblob import TextBlob

# def analyze_sentiment(text):
#     blob = TextBlob(text)
#     score = blob.sentiment.polarity  # Polarity score (-1 to 1)
    
#     if score > 0.2:
#         return "POSITIVE", score
#     elif score < -0.2:
#         return "NEGATIVE", score
#     else:
#         return "NEUTRAL", score

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return "POSITIVE", scores['compound']
    elif scores['compound'] <= -0.05:
        return "NEGATIVE", scores['compound']
    else:
        return "NEUTRAL", scores['compound']
