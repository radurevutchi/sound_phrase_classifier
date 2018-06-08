# sound_phrase_classifier
Classifies sound phrases from large scale corpora using NLP, POS tagging, and SMVs.

# Description
This project is a replication of the experiments conducted in Section 2 of the paper:
"Discovering sound concepts and acoustic relations in text" found on IEEE Xplore

The project processes large scale text corpora and uses regular expressions and POS tagging to classify sound phrases. I then manually labeled around 3000 sound phrases obtained previously into sound or non-sound classification. The resulting was used to train an SVM with an RBF kernel to produce a sound phrase vs non-sound phrase classifier.

The project runs in Python3.

# Files included
sounds.py (not used, but shows how training examples were discovered) <br/>
buildclassifier.py <br/>
newXy

# Files not included (must download)
Google's pretrained word2vec represantations model (found here: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
)

# Dependencies and Libraries
numpy (pip3 install -U numpy) <br/>
sklearn (pip3 install -U sklearn) <br/>
gensim (pip3 install -U gensim) <br/>
re (for sounds.py) <br/>
nltk (pip3 install -U nltk) (for sounds.py) <br/>
string (for sounds.py) <br/>


# How to Use

1. Download the repository
2. Download Google's pretrained word2vec representations
3. Run gzip -d on google's pretrained model file to turn it into a .bin file
4. Add path to the word2vec model in buildclassifier.py
5. Install dependencies numpy, sklearn, gensim
6. run: python3 buildclassifier.py
7. Input unigram or bigram phrases to test prediction
