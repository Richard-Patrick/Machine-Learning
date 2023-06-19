import pandas as pd
import nltk 
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras import regularizers





#data processing 
TweetsTrainData = pd.read_csv('Data/Tweets_train.csv',encoding = 'latin')
TweetsTestData = pd.read_csv('Data/Tweets_test.csv',encoding = 'latin')
TweetsDevData = pd.read_csv('Data/Tweets_dev.csv',encoding = 'latin')

print(TweetsTrainData.shape,TweetsDevData.shape,TweetsTestData.shape)

TweetsTrainData = TweetsTrainData.drop('tweet_id', axis=1)
TweetsTestData = TweetsTestData.drop('tweet_id', axis=1)
TweetsDevData = TweetsDevData.drop('tweet_id', axis=1)


#EDA

#dataset
print(TweetsTrainData.groupby('airline_sentiment').count())
print(TweetsTestData.groupby('airline_sentiment').count())
print(TweetsDevData.groupby('airline_sentiment').count())

#chart- train data
plot_target = TweetsTrainData.groupby('airline_sentiment').count().plot(kind='bar', title='Distribution of Train data',legend=False)
plot_target.set_xticklabels(['Negative','Positive','Neutral'], rotation=0)
#chart- test data
plot_target = TweetsTestData.groupby('airline_sentiment').count().plot(kind='bar', title='Distribution of Test data',legend=False)
plot_target.set_xticklabels(['Negative','Positive','Neutral'], rotation=0)
#chart- dev data
plot_target = TweetsDevData.groupby('airline_sentiment').count().plot(kind='bar', title='Distribution of Dev data',legend=False)
plot_target.set_xticklabels(['Negative','Positive','Neutral'], rotation=0)


#Data pre-processing
stop_words = stopwords.words('english')
stemmer = SnowballStemmer('english')

def preprocess(text, stem=False):
  text = re.sub("@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+", ' ', str(text).lower()).strip()
  tokens = []
  for token in text.split():
    if token not in stop_words:
      if stem:
        tokens.append(stemmer.stem(token))
      else:
        tokens.append(token)
  return " ".join(tokens)


TweetsTrainData.text =TweetsTrainData.text.apply(lambda x: preprocess(x))
TweetsTestData.text =TweetsTestData.text.apply(lambda x: preprocess(x))
TweetsDevData.text =TweetsDevData.text.apply(lambda x: preprocess(x))



temp = TweetsTrainData['text'].apply(lambda x:str(x).split())
top = Counter([item for sublist in temp for item in sublist])
temp = pd.DataFrame(top.most_common(20))
temp.columns = ['Common_words','count']
temp.style.background_gradient(cmap='Blues')


plt.barh(temp['Common_words'],temp['count'])
plt.xlabel("Count")
plt.ylabel("Common_words")
plt.title("Top common words in train data")


#nan remove
TweetsTrainData=TweetsTrainData[ (TweetsTrainData['text'].notnull()) & (TweetsTrainData['text']!=u'') ]
TweetsTestData=TweetsTestData[ (TweetsTestData['text'].notnull()) & (TweetsTestData['text']!=u'') ]
TweetsDevData=TweetsDevData[ (TweetsTrainData['text'].notnull()) & (TweetsDevData['text']!=u'') ]

#Split data
TweetsTrain_x=TweetsTrainData.text
TweetsTrain_y=TweetsTrainData.airline_sentiment
TweetsTest_x=TweetsTestData.text
TweetsTest_y=TweetsTestData.airline_sentiment
TweetsDevData_x=TweetsDevData.text
TweetsDevData_y=TweetsDevData.airline_sentiment


#Vectotizer
Vectorizer = TfidfVectorizer(max_features=6000)
Vectorizer.fit(TweetsTrain_x)
Train_Vectorizer_X = Vectorizer.transform(TweetsTrain_x)
Test_Vectorizer_X = Vectorizer.transform(TweetsTest_x)


#supervised learning
classifiers = [
    SVC(),
    MultinomialNB(),
    RandomForestClassifier(),
    GradientBoostingClassifier()
    ]
for classifier in classifiers:
    pipe = Pipeline(steps=[
                      ('classifier', classifier)])
    pipe.fit(Train_Vectorizer_X, TweetsTrain_y)   
    print(classifier,"model score: %.3f" % pipe.score(Test_Vectorizer_X, TweetsTest_y))
    print(classification_report(TweetsTest_y , pipe.predict(Test_Vectorizer_X)))









