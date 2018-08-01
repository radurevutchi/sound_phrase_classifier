from gensim.models import KeyedVectors
import numpy as np
from sklearn import svm
import sys, os
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import pickle



def vectorify(data, vector_model, glove):
    temp_X = []
    temp_y = []
    vectors = []

    # goes through all training/testing examples
    for example in data:

        #makes sure dataline is not empty
        if example:
            example = example.split(',')
            sound = example[0].split()
            correct = example[1]

            if len(sound) == 1:
                sound += sound
            assert(len(sound) == 2)


            # Makes lowercase
            if sound[0] not in vector_model:
                sound[0] = sound[0].lower()
            if sound[1] not in vector_model:
                sound[1] = sound[1].lower()


            #Makes sure at least one word is found in embeddings vector dict
            if sound[0] in vector_model or sound[1] in vector_model:
                if sound[0] in vector_model:
                    one = vector_model[sound[0]]

                #unknown word handling
                else:
                    one = vector_model['unk']

                if sound[1] in vector_model:
                    two = vector_model[sound[1]]

                #unknown word handling
                else:
                    two = vector_model['unk']

                #concatenates vectors together
                vector = list(one) + list(two)
                sound = ' '.join(sound)
                temp_y.append(int(correct))
                temp_X.append(sound)
                vectors.append(vector)


    return(temp_X, temp_y, vectors)








# Reads command line arguments
if len(sys.argv) != 5:
    sys.exit('Wrong number of arguments - ERROR')
else:
    if sys.argv[1] == 'glove':
        glove = True
    elif sys.argv[1] == 'word2vec':
        glove = False
    else:
        sys.exit("ERROR")

vectors_filename = sys.argv[2]
clf_filename = sys.argv[3]
test_filename = sys.argv[4]




# Loads the vector model (either word2vec or GloVe)
print("Loading vector embeddings...")
if glove:
    if os.path.isfile(vectors_filename+'.word2vec'):
        vector_model = KeyedVectors.load_word2vec_format(vectors_filename+'.word2vec',binary=False)
    else:
        glove2word2vec(vectors_filename, vectors_filename+'.word2vec')
        vector_model = KeyedVectors.load_word2vec_format(vectors_filename+'.word2vec',binary=False)
else:
    vector_model = KeyedVectors.load_word2vec_format(vectors_filename, binary=True)



# Loads the SVM classifier from file
print("Loading SVM sound classifier...")
clf = pickle.load(open(clf_filename, 'rb'))
#clf = joblib.load(clf_filename)



# Processes test data
print("Processing test data...")

test = open(test_filename,'r')
data = test.read().split('\n')






(X, y, vectors) = vectorify(data, vector_model, glove)
assert(len(X) == len(y) and len(X) == len(vectors))


num_sounds = len(X)
predictions = clf.predict(vectors)
accuracy = accuracy_score(y,predictions)

print('Accuracy for the test set on the given model is: ' + str(accuracy))
