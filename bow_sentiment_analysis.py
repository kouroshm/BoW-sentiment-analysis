import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier

# Preprocessing
OUTPUT_FOLDER = 'input the directory of your outputfolder'
data = pd.read_csv('/data/reviews.csv', delimiter='\t', header=None)
data.rename(columns={0: 'Company Name', 1: 'Rating', 2: 'Date Published',
                        3: 'Review Body'}, inplace=True)
data.drop(index=data.index[0], 
        axis=0, 
        inplace=True)
print('Number of rows per rating: ')
print(data['Rating'].value_counts())


def map_sentiment(rating):

    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


data['sentiment'] = [map_sentiment(int(i)) for i in data['Rating']]


def top_data(n_samples):
    top_positive = data[data['sentiment'] == 2].head(n_samples)
    top_neutral = data[data['sentiment'] == 1].head(n_samples)
    top_negative = data[data['sentiment'] == 0].head(n_samples)
    data_bal = pd.concat([top_positive, top_neutral, top_negative])
    return data_bal


# Split the training and validation and save it
data_bal = top_data(n_samples=175)
data_bal = data_bal.drop(columns=['Company Name', 'Rating', 'Date Published'])
train, val = train_test_split(data_bal, test_size=0.2, random_state=0)
train.to_csv(OUTPUT_FOLDER + '/training.csv', index=False)
val.to_csv(OUTPUT_FOLDER + '/validation.csv', index=False)
# Load training.csv and train the model
train = pd.read_csv(OUTPUT_FOLDER + '/training.csv')
X_train = train['Review Body']
Y_train = train['sentiment']
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer(use_idf=True).fit(train_counts)
train_tf = tf_transformer.transform(train_counts)
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
# After validation SGDClassifier was the best model
sgd_clf = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, random_state=42,
                        max_iter=5, tol=None).fit(train_tfidf, Y_train)
# you can change it to test.csv
val = pd.read_csv(OUTPUT_FOLDER + '/validation.csv')
docs_test = val['Review Body']
X_new_counts = count_vect.transform(docs_test)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted_sgd = sgd_clf.predict(X_new_tfidf)
print(' ')
print('Accuracy of SGD: ', accuracy_score(val['sentiment'], predicted_sgd))
print('F1 Score of SGD: ', f1_score(val['sentiment'], predicted_sgd,
      average='weighted'))
print(' ')
print('Confusion Matrix: ')
cmtx = pd.DataFrame((confusion_matrix(val['sentiment'], predicted_sgd)),
                    index=['negative', 'neutral', 'positive'],
                    columns=['negative', 'neutral', 'positive'])
print(cmtx)
print(' ')
print('Classification Report: ')
print(classification_report(val['sentiment'], predicted_sgd))
# The biggest problem is when we downsample
# the data, number of entries will decrease
# Therefore, it is important for the dataset
# to be bigger for more optimal detection
# in neutral sector of ratings
