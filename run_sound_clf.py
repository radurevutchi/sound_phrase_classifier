from gensim.models import KeyedVectors
import numpy as np
from sklearn import svm
import random, nltk, sys, os, re
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.externals import joblib
import pickle


def find_vector(sound, vector_model):

    sound = sound.split()
    assert(len(sound) == 1 or len(sound) == 2)

    if len(sound) == 1:
        if sound[0] in vector_model:
            result = vector_model[sound[0]]
            result = list(result)
            result += result
        elif sound[0].lower() in vector_model:
            result = vector_model[sound[0].lower()]
            result = list(result)
            result += result
        else:
            result = []


    else:

        if '_'.join(sound) in vector_model:
            result = vector_model['_'.join(sound)]
            result = list(result)
            result += result
        elif '_'.join(sound).lower() in vector_model:
            result = vector_model['_'.join(sound).lower()]
            result = list(result)
            result += result
        elif sound[0] in vector_model or sound[1] in vector_model:
            if sound[0] in vector_model:
                one = vector_model[sound[0]]
            elif sound[0].lower() in vector_model:
                one = vector_model[sound[0].lower()]
            else:
                one = vector_model['unk']
            if sound[1] in vector_model:
                two = vector_model[sound[1]]
            elif sound[1].lower() in vector_model:
                two = vector_model[sound[1].lower()]
            else:
                two = vector_model['unk']
            one = list(one)
            two = list(two)
            result = one + two
        else:
            result = []


    return result


def process_POS(sound):
    punctuation = '!"#$%&()*+,-./:;<=>?@[\\]^_{|}~'

    tokens = nltk.word_tokenize(sound)
    tokens = nltk.pos_tag(tokens)

    justtags = map(lambda x: x[1], tokens)
    justtags = list(justtags)

    phraselen = 0

    #unigram was given as input
    if len(justtags) == 1:
        if justtags[0] == 'VBG':
            phraselen = 1
        elif justtags[0] == 'NN' or justtags[0] == 'NNS':
            phraselen = 1



    #bigram was given as input
    else:
        if justtags[0] == 'VBG':
            if justtags[1] == 'NN' or justtags[1] == 'NNS':
                phraselen = 2 #(DT) VBG NN(S)

        elif justtags[0] == 'NN' or justtags[0] == 'NNS':
            if (justtags[1] =='NN' or justtags[1] == 'NNS') and justtags[0] == 'NN':
                phraselen = 2 # (DT) NN NN(S)
            elif justtags[1] == 'VBG':
                phraselen = 2 # (DT) NN(S) VBG

        elif justtags[0] == 'JJ':
            if justtags[1] =='NN' or justtags[1] == 'NNS':
                phraselen = 2 # (DT) JJ NN(S)



    sound = ' '.join(sound.split()[:phraselen])
    translator = str.maketrans('','',punctuation)
    sound = sound.translate(translator)

    return sound




def process_doc(filename):
    temp_data = []
    raw_file = open(filename, 'r')
    data = ' '.join(raw_file.read().split())
    findings = re.findall('sounds?\sof\s(\S+?\s\S+?)\s', data)

    return findings



def vectorify(data, vector_model, clf):
    temp_data = []
    vectors = []
    scores = []
    bigrams = []
    unigrams = []

    count = 0
    a = open(data, 'r')
    example = a.readline()[:-1]
    # goes through all training/testing examples
    while example:
        print(count)

        #makes sure dataline is not empty
        if example:
            example = example.split()


            #bigram checking

            if len(example) > 1:
                for i in range(len(example)-1):
                    sound = process_POS(' '.join(example[i:i+1]))
                    if sound: bigrams.append(sound)

            #unigram checking
            for word in example:
                sound = process_POS(word)
                if sound: unigrams.append(sound)




            # Makes lowercase
            for bigram in bigrams:
                vector = find_vector(bigram, vector_model)
                if vector: vectors.append(vector)


            for unigram in unigrams:
                vector = find_vector(unigram, vector_model)
                if vector: vectors.append(vector)



            if vectors:
                confidence = clf.decision_function(vectors)
                max_confidence = max(confidence)
                if max_confidence > 0:
                    example = ' '.join(example)
                    temp_data.append(example)
                    scores.append(max_confidence)
        count += 1
        example = a.readline()[:-1]
        vectors = []
        bigrams = []
        unigrams = []


    return(temp_data, scores)








# Reads command line arguments
if len(sys.argv) != 6:
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
sentences = sys.argv[5].lower()

if sentences == 'true':
    sentences = True
elif sentences == 'false':
    sentences = False
else:
    sys.exit('ERROR - check data filetype (True/False)')





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



# Processes test data
print("Processing test data...")
if sentences:
    data = process_doc(test_filename)
else:
    test = open(test_filename,'r')
    data = test.read().split('\n')




(data,scores) = vectorify(test_filename, vector_model, clf)

assert(len(data) == len(scores))


output_filename = 'results.txt'
output = open(output_filename, 'w')
for index in range(len(data)):
    output.write(data[index] + ',' + str(scores[index]) + '\n')

output.close()


print('List of sounds can be found in ' + output_filename)
