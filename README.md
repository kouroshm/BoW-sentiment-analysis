# Sentiment Analysis Using Bag of Words Method
In this project I have done a sentiment analysis using BoW method on Trustpilot reviews.

# Usage
As mentioned in [Web Scraping](https://github.com/kouroshm/web-scraping) repository we used Bag of Words method to do sentiment analysis on reviews that was submitted on Trustpilot for the company Playa Cabana. As long as the sentiment analysis is done on the Trustpilot you can scrape it using [Web Scraping](https://github.com/kouroshm/web-scraping) and use this method to do sentiment analysis. However, this method can be extended to wider variety of analysis such as product review analysis, movie analysis, and etc.

# Installation
Run the following command to install all the necessary library requirements:  
`pip install -r requirements.txt`

You have to input an output folder to save the training and validation data for training the SGDClassifier using the training data by using TFIDF transformer to to tokenize the words and count the occurance of them. The following line is the code to change:    
`OUTPUT_FOLDER = 'input the directory of your outputfolder'`