

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
# Download NLTK resources if not already downloaded
nltk.download('vader_lexicon')
sid = SentimentIntensityAnalyzer()
def sentiment_analysis(text):
    # Get the sentiment scores for the text
    scores = sid.polarity_scores(text)
    # Determine the overall sentiment based on the compound score
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

amazon_reviews=[]
sentiments=[]
 
import csv
with open('amazon_dataset.csv') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)      
    #f = open("amazon_reviews_sentiment.csv", "a")  
    for row in reader:
        print(row[2])
        #amazon_reviews.append(row[2])
        sentiments.append(sentiment_analysis(row[2]))
        #f.write(sentiment+"\n")
    #f.close()
#sentiments=[]
#for i in range (0, 15000):
#    sentiment = sentiment_analysis(amazon_reviews[i])
#    sentiments.append(sentiment)
#    print(sentiment)

import matplotlib.pyplot as plt
from collections import Counter

# Count the frequency of each string
string_counts = Counter(sentiments)

# Extract strings and their corresponding counts
labels, counts = zip(*string_counts.items())

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(labels, counts, color='skyblue')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.title('Sentiment Analysis for '+str(len(sentiments))+' generated reviews')
plt.xticks(rotation=45)   
plt.tight_layout()   
#plt.show()
plt.savefig('sentiment.png')