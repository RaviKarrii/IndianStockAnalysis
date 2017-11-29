import pandas as pd
import quandl
import tweepy
import requests
import numpy as np
from matplotlib import pyplot as plt
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense

FILE_NAME = 'Data/historical.csv'
# First we login into twitter
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
user = tweepy.API(auth)

def stock_sentiment(quote, num_tweets):
    # Checks if the sentiment for our quote is
    # positive or negative, returns True if
    # majority of valid tweets have positive sentiment
    list_of_tweets = user.search(quote, count=num_tweets)
    positive, null = 0, 0
    for tweet in list_of_tweets:
        blob = TextBlob(tweet.text).sentiment
        if blob.subjectivity == 0:
            null += 1
            next
        if blob.polarity > 0:
            positive += 1

    if positive > ((num_tweets - null)/2):
        return "Positive"
    else:
        return "Negative"

def stock_prediction():

    # Collect data points from csv
    dataset = []

    with open(FILE_NAME) as f:
        for n, line in enumerate(f):
            if n != 0:
                dataset.append(float(line.split(',')[5]))
    dataset.reverse()
    dataset = np.array(dataset)
    # Create dataset matrix (X=t and Y=t+1)
    def create_dataset(dataset):
        dataX = [dataset[n+1] for n in range(len(dataset)-2)]
        return np.array(dataX), dataset[2:]
        
    trainX, trainY = create_dataset(dataset)

    # Create and fit Multilinear Perceptron model
    model = Sequential()
    model.add(Dense(8, input_dim=1, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

    # Our prediction for tomorrow
    prediction = model.predict(np.array([dataset[0]]))
    result = 'The price may move from %s to %s Tommorow' % (dataset[0], prediction[0][0])

    return result

    

"""writer = pd.ExcelWriter('StockData/Stock.xlsx')"""
def getstockdatafromQuandl(Stock,plotting = False):
    Stock1 = "NSE/"+ Stock
    print "Getting Stock history data for "+ Stock
    r = quandl.get(Stock1, rows=90)
    print "Query Succesful"
    print "Writing data to CSV"
    r.to_csv(FILE_NAME)
    """r.to_excel(writer,Stock)"""
    print "Writing Done"
    if plotting == True:
        plt.plot(r.index,r.Close)
        plt.title(Stock)
        plt.xlabel("Date", fontsize=8)  
        plt.ylabel("Close", fontsize=8)
        plt.grid(True)
        plt.show()

m = pd.read_csv("Data/Stock.csv")
for i in m.STOCK:
    Stock = i
    getstockdatafromQuandl(i)
    print stock_prediction()
    print i + " is a "+stock_sentiment(i,100)+" Stock"
