import re, nltk, string, glob, os



#Opens a text file and clears it of any html tags
def processfile(direction):

    current_read = open(direction, 'rb')
    raw_data = current_read.read().decode('ISO-8859-1')
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_data)
    return ' '.join(raw_data.split())



#Finds all 4-word phrases beginning with sound(s) of xx xx xx xx
def findsounds(filename):
    new_data = processfile(filename)
    pattern = 'sounds?\sof((\s\S+){4})'
    results = re.findall(pattern, new_data)
    results = map(lambda x: x[0], results)
    return results


#Classifies phrase into one of 6 POS tags or returns false
def checktagger(tags):


    justtags = map(lambda x: x[1], tags)
    justtags = list(justtags)

    if justtags[0] == 'VBG':
        if justtags[1] == 'NN' or justtags[1] == 'NNS':
            return (True, 2) #(DT) VBG NN(S)
        else:
            return (True, 1) # VBG

    elif justtags[0] == 'NN' or justtags[0] == 'NNS':
        if (justtags[1] =='NN' or justtags[1] == 'NNS') and justtags[0] == 'NN':
            return (True, 2) # (DT) NN NN(S)
        elif justtags[1] == 'VBG':
            return (True, 2) # (DT) NN(S) VBG
        else:
            return (True, 1) # (DT) NN(S)

    elif justtags[0] == 'JJ':
        if justtags[1] =='NN' or justtags[1] == 'NNS':
            return (True, 2) # (DT) JJ NN(S)

    return (False, 0) # did not match



allphrases = []

# tokenizes phrases and does POS tagging
def getsoundlist(directory, num_files):

    allphrases = []
    for i in range(num_files):

        phrases = findsounds(directory + str(i))
        for phrase in phrases:

            try:

                splitphrase = nltk.word_tokenize(phrase)
                tagger = nltk.pos_tag(splitphrase)

                if tagger[0][1] == 'DT':
                    tagger.pop(0)
                    splitphrase.pop(0)

                correct_fit = checktagger(tagger)
                if correct_fit[0]:
                    good_fit = ' '.join(splitphrase[0:correct_fit[1]])
                    good_fit = good_fit.lower()
                    translator = str.maketrans('','',string.punctuation)
                    good_fit = good_fit.translate(translator)
                    allphrases.append(good_fit)


            except:
                pass

    return allphrases

# top level function
def usemydata():
    results = getsoundlist('../Downloads/blogs/', 19320)
    results += getsoundlist('../Downloads/Gutenberg/txt/', 3036)
    results =  list(set(results))

    return results






# renames all files in this directory to numbers
path1 = '/home/radu/Downloads/Gutenberg/txt'
count = 0
for filename in glob.iglob(os.path.join(path1, '*.txt')):
    title, ext = os.path.splitext(os.path.basename(filename))
    os.rename(filename, os.path.join(path1, str(count)))
    count += 1




# renames all files in this directory to numbers
path1 = '/home/radu/Downloads/blogs'
count = 0
for filename in glob.iglob(os.path.join(path1, '*.txt')):
    title, ext = os.path.splitext(os.path.basename(filename))
    os.rename(filename, os.path.join(path1, str(count)))
    count += 1


dataset = usemydata()


# uncomment to write results to file

'''
X = open('X', 'w')

for phrase in dataset:
    X.write(phrase + '\n')
X.close()
'''
