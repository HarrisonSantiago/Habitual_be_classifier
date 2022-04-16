import numpy as np
from nltk import *
from Habitual_be_classifier.Train import *
from Habitual_be_classifier.Augmenter import augmenter
from sklearn.feature_extraction.text import CountVectorizer


def vectorize(dataset):

    """
    Takes in the the unknown habituality instances after rule-based filtering
    and returns the input for the ML training

    Parameters
        - a dataset of [samples]
    Returns
        - and array of size [samples x features] where features are the available POS tags

    """

    X = dataset[:,0]

    #Now I need to vectorize the data
    countfeatures = []
    for row in X:

        #Created a tokenized list of each sentence
        tokenizer = RegexpTokenizer(r"\w+")
        tokenized = tokenizer.tokenize(row)
        token_list = pos_tag(tokenized)

        location_of_be = 0

        #Creates a list of strings that consists solely of the part of speech tags around "be"
        for i in range(8, len(token_list)-5):

            if token_list[i][0]=="be" or token_list[i][0]=="Be":
                location_of_be = i
                break
        countToAdd = ""
        for i in range(location_of_be-6, location_of_be+5):
            countToAdd = countToAdd + token_list[i][1]+" "

        countfeatures.append(countToAdd)

    #Makes a count vector of the list
    #i.e. if count features = ["NN VB VB ADV" , "VB ADJ JJ"]
    #then X_counts = [ [1 2 1 0 0] , [0 1 0 1 1] ]
    sent = "CC CD DT EX FW IN JJ IN JJ JJR JJS LS MD NN NNS NNP NNPS PDT POS PRP PPS" \
           " RB RBR RBS RP TO UH VB VBG VBD VBN VBP VBZ WDT WP WRB"

    countfeatures.insert(0, sent)

    counter_vect = CountVectorizer()
    X_counts = counter_vect.fit_transform(countfeatures)
    X_counts = np.delete(X_counts.toarray(), (0), axis=0)
    return X_counts


def get_classifiers(unknown_hab):

    """
    Gets the array from processor and returns the trained ML models

    Parameters
        - unknown_hab from rule filter
    Returns
        - dictionary with the trained classifier models

    """

    augmented = augmenter(unknown_hab)

    unknown_hab = np.concatenate((unknown_hab, augmented), axis = 0)

    y = unknown_hab[:,2].astype(np.int)

    X = vectorize(unknown_hab)

    classifiers = train_models(X, y)

    return classifiers


