# 506-final-project

## Description
News Vs. Stock price analysis. Based on recent articles from sources such as NYT, etc. how does one article result in stock price increase or decrease.


## Clear Goal(s)

To analyze how news articles from sources like The New York Times impact stock price fluctuations. Specifically, the project aims to determine whether an article leads to a stock price increase or decrease based on sentiment analysis.


## Data Collection and Methods

The data that needs to be collected are articles that come out from past and present, so that we can analyze whether they are positive and negative. We can use an API or web scrape articles across different new outlets and use another API like Yahoo Finance to see a stock’s price change at the time of the article’s release. 


## Data Modelling

First, we’d need to use a NLP model to more accurately understand the meaning behind an article. Then, we plan on modeling the data through a linear regression model that associates the degree of positivity/negativity with past occurrences of increase/decrease in stock price


## Data Visualization

We plan on using scatter plots to plot the sentiment scores versus the stock price variations to view changes. Additionally, we would have a time series plot to view the stock price fluctuations while marking article release. Finally, we would have heatmaps to view the correlation matrix between sentiment and stock price changes.

## Test Plan
To ensure reliability in our model we will implement:
1) Train test split: divides the dataset into 80 percent training data and 20 percent testing data to train the model on the training set and test the performance on unseen testing set of 20 percent to evaluate adaptability to new info.
2) Time split: we train the model on past data in previous months and test it on future data in the upcoming months to predict the stock movements.
3) k fold: we divide into folds and train model on k-1 folds and the remaining kth fold is used for testing. We would repeat each k time and rotate the fold to be different to prevent overfitting.
