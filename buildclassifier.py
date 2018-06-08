from gensim.models import KeyedVectors
import numpy as np
from sklearn import svm


# takes in bigram or
def prediction(model, classifier, phrase):
    phrases = phrase.split()
    try:
        if len(phrases) == 2:
            a = model[phrases[0]] # 300 dimensional vector
            b = model[phrases[1]] # 300 dimensional vector
            data = np.array([a,b])
            average = np.average(data, axis=0) #takes average of 2 vectors
            average = list(average)
        elif len(phrases) == 1:
            print('unigram')
            average = model(phrases[0])
        else:
            return [0]  # trigram or greater phrase
        return classifier.predict([average]) # outputs prediction


    except:
        return [0] # the phrase is not in google pretrained word2vec model


# reads training data
Xy = open('newXy', 'r')
sound_phrase = Xy.readline()
dataset = []

while(sound_phrase):
    dataset.append(sound_phrase)
    sound_phrase = Xy.readline()

mydata = []

for index in range(len(dataset)):
    dataset[index] = ''.join(dataset[index].split(','))
    mydata = mydata + dataset[index].split()

# splits dataset into X and y for training
X = []
y = []

for a in range(2,len(mydata),3):
    y.append(int(mydata[a]))

for a in range(0,len(mydata),3):
    X.append(mydata[a] + ' ' + mydata[a+1])




assert(len(X) == len(y))





print('Loading pretrained word2vec representations...')

#################################################
#CHANGE PATH TO GOOGLE'S PRETRAINED VECTORS HERE#
#################################################
model = KeyedVectors.load_word2vec_format('./google.bin', binary=True)

failed = 0
features = []

print('Transforming phrases into word2vec vectors...')

for phrase in range(len(X)-1, -1, -1):
    phrases = X[phrase].split()

    # loads training features from bigrams stored in X
    try:
        a = model[phrases[0]]
        b = model[phrases[1]]
        data = np.array([a,b])
        average = np.average(data, axis=0)
        features.append(list(average))

    except: # error in converting to word2vec representation
        failed += 1
        y.pop(phrase)
        X.pop(phrase)


print('Building the SVM classifier...')

# uses first 2800 training examples for training

X = X[:2800]
features = features[:2800]
y = y[:2800]


assert(len(features) == len(y))




# trains SVM with RBF kernel
# trains on 2800 training examples
# gamma and C values found using optunity library with Ankit's help

clf = svm.SVC(C=10 ** 1.9817533052884615, gamma=10 ** -2.3008980049838534).fit(features, y)


print('Running original training data on SVM model...')

good = 0
bad = 0

for index in range(len(X)):
    a = prediction(model, clf, X[index])
    a = a[0]
    if a == y[index] and a == 1:
        #print(X[index]) # Uncomment for list of sound phrases according
                        # to the model from the training data
        good += 1 # correctly classified
    else:
        bad += 1 # incorrectly classified


print('# Correctly Classified: ' + str(good))
print('# Incorrectly Classified: ' + str(bad))
print()


# Unigram and Bigram prediction from user input
print('Enter a unigram or bigram phrase to predict: ("stop" to quit program)')
newprediction = input()
while newprediction != 'stop':

    a = prediction(model, clf, newprediction)
    if a == [1]:
        print('This is a sound.')
        print()
    else:
        print('This is NOT a sound')
        print()

        
    print('enter a phrase to predict')
    newprediction = input()
