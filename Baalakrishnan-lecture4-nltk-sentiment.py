import csv
import matplotlib.pyplot as plt
from nltk.sentiment.util import demo_liu_hu_lexicon as lhl
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import opinion_lexicon
from nltk.tokenize import treebank


def _show_plot(x_values, y_values, x_labels=None, y_labels=None):
    plt.locator_params(axis="y", nbins=3)
    axes = plt.axes()
    axes.yaxis.grid()
    plt.plot(x_values, y_values, "ro", color="red")
    plt.ylim(ymin=-1.2, ymax=1.2)
    plt.tight_layout(pad=5)
    if x_labels:
        plt.xticks(x_values, x_labels, rotation="vertical")
    if y_labels:
        plt.yticks([-1, 0, 1], y_labels, rotation="horizontal")
    # Pad margins so that markers are not clipped by the axes
    plt.margins(0.2)
    plt.show()


""" This function(below) uses the Liu and Hu Opinion Lexicon provided in NLTK to find the polarity of each word in the tweet. 
The code tries to calculate the arrive at the overall sentiment of the tweet by counting on the number of Positive and 
negative words in the sentence.
The opinion lexicon contains a list of words with their polarities in three categories : Postive, negative and neutral
if a word is not part of the lexicon , it is assumed to be neutral 
When the number of positive words are higher than that of Negative words then the overall
sentiment of the tweet is positive. Similarly, if negative words are greater than positive then it returns Negative as the
overall sentiment.
This code has been derived from the demo code in sentiment.util package
"""

def liu_hu_sentiment(sentence, plot=False):
    tokenizer = treebank.TreebankWordTokenizer()
    pos_words = 0
    neg_words = 0
    tokenized_sent = [word.lower() for word in tokenizer.tokenize(sentence)]

    x = list(range(len(tokenized_sent)))  # x axis for the plot
    y = []

    for word in tokenized_sent:
        if word in opinion_lexicon.positive():
            pos_words += 1
            y.append(1)  # positive
        elif word in opinion_lexicon.negative():
            neg_words += 1
            y.append(-1)  # negative
        else:
             y.append(0)  # neutral

    if pos_words > neg_words:
        output= "Positive"
    elif pos_words < neg_words:
        output= "Negative"
    elif pos_words == neg_words:
        output= "Neutral"
    if plot == True:
        _show_plot(x, y, x_labels=tokenized_sent, y_labels=["Negative", "Neutral", "Positive"])
    return output

# Reading data from the CSV file
data = list(csv.reader(open(r'F:\dataset_sentiment_analysis.csv'),delimiter=','))
nike_tweets=[]
# Collecting all tweets which are of the topic "Nike"
for i in range(len(data)):
    if data[i][3] == "nike":
        nike_tweets.append(liu_hu_sentiment(data[i][5]))    # Calls the liu_hu_sentiment function which returns the overall polarity of the tweet
print(nike_tweets)
print("Positive : %f"%((nike_tweets.count('Positive')/len(nike_tweets))*100))
print("Negative : %f"%((nike_tweets.count('Negative')/len(nike_tweets))*100))
print("Neutral : %f"%((nike_tweets.count('Neutral')/len(nike_tweets))*100))

# To idenitfy tweets of World Cup 2010 and plot sentence polarities
for i in range(len(data)):
    if data[i][3] == "world cup 2010":
        liu_hu_sentiment(data[i][5], plot=True)


# Dentist tweets are collected and Vader approach is used to calculate the Average Compound Polarity score
dentist_tweets=[]

for i in range(len(data)):
    if data[i][3] == "dentist":
        dentist_tweets.append(data[i][5])

"""VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool that is 
specifically attuned to sentiments expressed in social media. VADER uses a combination of A sentiment lexicon is a list 
of lexical features (e.g., words) which are generally labelled according to their semantic orientation as either positive 
or negative.
The Compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized 
between -1(most extreme negative) and +1 (most extreme positive)."""

sid = SentimentIntensityAnalyzer()
for sentence in dentist_tweets:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
        print()



