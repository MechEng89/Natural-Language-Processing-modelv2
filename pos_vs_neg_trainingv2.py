import nltk
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
import random

documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

stop_words=set(stopwords.words("english"))

all_words = []
filter_all_words =[]
words=[]

#for w in movie_reviews.words():
#    all_words.append(w.lower())

for w in movie_reviews.words():
    if w not in stop_words:
        filter_all_words.append(w.lower())
        
filter_all_words = nltk.FreqDist(filter_all_words)

#print ("top 20 common words")
#print (filter_all_words.most_common(20))
#print ("stop words")
#print (stop_words)

word_features = list(filter_all_words.keys())[:5000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
            features[w] = (w in words)  # compare the whole list of words with ranked classifer
    return features
    
    
#print ((find_features(movie_reviews.words("neg/cv000_29416.txt"))))
#print (word_features)

featuresets = [(find_features(rev), category) for (rev,category) in documents]

training_set = featuresets[:1500]
testing_set = featuresets[1500:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print ("Native Bayes Alogrium accurancy precentage:",(nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(20)
