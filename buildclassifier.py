from gensim.models import KeyedVectors
import numpy as np
from sklearn import svm
import random



# HELPER FUNCTIONS


# takes in bigram or unigram and returns whether it is a sound
def prediction(model, classifier, phrase):
    phrases = phrase.split()
    try:
        if len(phrases) == 2:
            a = model[phrases[0]] # 300 dimensional vector
            b = model[phrases[1]] # 300 dimensional vector
            a = list(a)
            b = list(b)
            average = a + b # 600 dimensional vector
        elif len(phrases) == 1:
            a = model[phrases[0]]
            a = list(a)
            average = a + a

        else:
            return [0]  # trigram or greater phrase


        return classifier.predict([average]) # outputs prediction


    except:
        return [0] # the phrase is not in google pretrained word2vec model

def printaccuracy(model, classifier, X, y, text='Dataset Accuracy'):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for index in range(len(X)):
        a = prediction(model, clf, X[index])
        a = a[0]
        if a == y[index] and a == 1:
            tp += 1
        elif a == y[index] and a == 0:
            tn += 1
        elif a != y[index] and a == 0:
            fn += 1
        else:
            fp += 1

    print('Accuracy for: ' + text)
    print('tp: ' + str(tp))
    print('tn: ' + str(tn))
    print('fp: ' + str(fp))
    print('fn: ' + str(fn))

    print('accuracy: ' + str((tp + tn)/(tp+tn+fp+fn)))
    #print('built-in accuracy: ' + str(clf.score(features_test, y_test)))

    print()
    print()


# reads training data from file

def readdata(directory):
    training_file = open(directory, 'r')
    sound_phrase = training_file.readline()
    dataset = []

    while(sound_phrase):
        dataset.append(sound_phrase)
        sound_phrase = training_file.readline()

    random.shuffle(dataset)

    cleandata = []

    for index in range(len(dataset)):
        dataset[index] = ''.join(dataset[index].split(','))
        cleandata = cleandata + dataset[index].split()

    # splits dataset into X and y for training
    X = []
    y = []

    for a in range(2,len(cleandata),3):
        y.append(int(cleandata[a]))

    for a in range(0,len(cleandata),3):
        X.append(cleandata[a] + ' ' + cleandata[a+1])

    return [X,y]


# Allows user to test unigrams and bigrams on classifier
def userinputprediction(google, clf):

    print('Enter a unigram or bigram phrase to predict: ("stop" to quit program)')
    newprediction = input()
    while newprediction != 'stop':

        a = prediction(google, clf, newprediction)
        if a == [1]:
            print('This is a sound.')
            print()
        else:
            print('This is NOT a sound')
            print()


        print('enter a phrase to predict')
        newprediction = input()




# MAIN CODE


# reads file
print('Reading data from file...')

data = readdata('new_training_data') # path to data file
X = data[0] # bigram phrases
y = data[1] # labels( 0s and 1s)
features = []

assert(len(X) == len(y))





print('Loading pretrained word2vec representations...')

# model which converts words to word2vec representations
google = KeyedVectors.load_word2vec_format('./google.bin', binary=True)







print('Transforming phrases into word2vec vectors...')

failed = 0 # phrases for which google doesn't have a word2vec representation
for phrase in range(len(X)-1, -1, -1):
    phrases = X[phrase].split()

    # loads training features from bigrams stored in X
    try:
        a = google[phrases[0]]
        b = google[phrases[1]]
        a = list(a)
        b = list(b)
        average = a + b # concatenates the two vectors
        features = [average] + features # adds it to list of features

    except: # error in converting to word2vec representation
        failed += 1 # number of failed word2vec translations
        y.pop(phrase)
        X.pop(phrase)







print('Building the SVM classifier...')
print()
print()


m = len(X) # number of examples in data
split = 4700 # number to split training and test data between
assert(0 < split and split < m)

# [split:m] examples are used for test data
X_test = X[split:]
features_test = features[split:]
y_test = y[split:]

# split examples are used for training data
X = X[:split]
features = features[:split]
y = y[:split]


assert(len(features) == len(y))


print('Training on ' + str(split) + ' examples.')
print('Testing on ' + str(m-split) + ' examples.')
print()
print()
# Value of C found using optunity library
clf = svm.LinearSVC(C=10 ** -1.7716676259261839, dual=False)
clf.fit(features, y)

# Accuracy run on training data and on test data
printaccuracy(google, clf, X, y, text='Training Set')
printaccuracy(google, clf, X_test, y_test, text='Test Set')






print('Running original training data on SVM model...')


# Unigram and Bigram prediction from user input
userinputprediction(google, clf)
