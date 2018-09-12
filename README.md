# sound_phrase_classifier
Classifies sound phrases from large scale corpora using NLP, POS tagging, Word Embeddings, and SVMs.

# Description
This project is a replication of the experiments conducted in Section 2 of the paper:
"Discovering sound concepts and acoustic relations in text" found on IEEE Xplore </br>

The project processes large scale text corpora and uses regular expressions and POS tagging to classify sound phrases. I then manually labeled around 3000 sound phrases obtained previously into sound or non-sound classification. The resulting was used to train a Linear SVM to produce a sound phrase vs non-sound phrase classifier.<br/>


The project runs in Python3.

# Files included
train_sound_clf.py <br/>
test_sound_clf.py <br/>
run_sound_clf.py<br/>
training_data<br/>
<br/>
Additional Files:<br/>
training_data (training data for train_sound_clf.py)<br/>
clf1.model (classifier model trained on word2vec 300d vectors)<br/>
sample_document (input for run_sound_clf.py when set to "true")<br/>
sample_list (input  for run_sound_clf.py when set to "false")<br/>
results.txt (output from run_sound_clf.py when input is sample_list)<br/>

# Files not included (must download)
Google's pretrained word2vec represantations model (found here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
)<br/>
Stanford's GloVe pretrained vectors (found here: https://nlp.stanford.edu/projects/glove/)

# Dependencies and Libraries
numpy, optunity, gensim, sklearn, pickle, sys, os, nltk, re


# How to Use

To train the sound_classifier on new data and get a saved copy of the LinearSVM model, run:<br/>
python3 train_sound_clf.py (word2vec/glove) <embeddings_filename> <training_data_filename> <br/>
This will save the classifier model to the filename 'clf1.model'
<br/>
<br/>


To test the accuracy of the sound classifier on a list of labeled data, run:<br/>
python3 test_sound_clf.py (word2vec/glove) <embeddings_filename> <model_filename> <test_data_filename><br/>
This will print the accuracy of the classifier on the test data.<br/>

<br/>
<br/>

To run the classifier on a large text document or a list of unlabeled sounds, run:<br/>
python3 run_sound_clf.py (word2vec/glove) <embeddings_filename> <model_filename> <data_filename> (true/false)<br/>
(True for large document, false for list of sounds)
This will process the document(or list) and output a list (results.txt) of filtered sound phrases with their confidence scores.<br/>

<br/>
<br/>

IMPORTANT: A classifier may be trained on glove or word2vec embeddings only. Additionally, the input files for training_data and sample_list (when run_sound_clf.py set to 'false') must match the format given in the examples files.
