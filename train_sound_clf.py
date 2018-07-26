from gensim.models import KeyedVectors
import numpy as np
from sklearn import svm
import sys, os
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.externals import joblib
import optunity
import optunity.metrics
import pickle





def vectorify(data, vector_model):
    temp_X = []
    temp_y = []
    vectors = []
    for example in data:
        if example:
            #print(example)
            example = example.split(' , ')
            #print(example)
            sound = example[0].split()
            correct = example[1]
        try:
            if len(sound) == 2:
                one = vector_model[sound[0]]
                two = vector_model[sound[1]]
            else:
                one = vector_model[sound[0]]
                two = vector_model[sound[0]]
            one = list(one)
            two = list(two)
            vector = one + two
            sound = ' '.join(sound)
            temp_y.append(int(correct))
            temp_X.append(sound)
            vectors.append(vector)
        except:
            pass

    return(temp_X, temp_y, vectors)








# Reads command line arguments
if len(sys.argv) != 4:
    sys.exit('Wrong number of arguments - ERROR')
else:
    if sys.argv[1] == 'glove':
        glove = True
    elif sys.argv[1] == 'word2vec':
        glove = False
    else:
        sys.exit("ERROR")

vectors_filename = sys.argv[2]
training_filename = sys.argv[3]




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





# Processes test data
print("Processing test data...")

test = open(training_filename,'r')
data = test.read().split('\n')

(X, y, vectors) = vectorify(data, vector_model)
assert(len(X) == len(y) and len(X) == len(vectors))




# Training the model using optunity
print('Training the LinearSVM...')
@optunity.cross_validated(x=vectors, y=y, num_folds=2, num_iter=1)
def svm_auc(x_train, y_train, x_test, y_test, logC):
    model = svm.LinearSVC(C=10 ** logC).fit(x_train, y_train)
    decision_values = model.decision_function(x_test)
    return optunity.metrics.roc_auc(y_test, decision_values)

# perform tuning
hps, _, _ = optunity.maximize(svm_auc, num_evals=200, logC=[-4, 2])

# train model on the full training set with tuned hyperparameters
optimal_model = svm.LinearSVC(C=10 ** hps['logC']).fit(vectors, y)

print('The model has been dumped to the file: clf1.model')
pickle.dump(optimal_model, open('clf1.model', 'wb'))
