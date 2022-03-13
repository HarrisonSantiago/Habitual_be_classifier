from nltk import *
import numpy as np



def rule_filter(dataset):

    """
    Rule based filter
    Parameters
        - Dataset as read by csv_processor
    Returns
        - Array of "be" instances label as non-habitual
        - Array of "be" instances which could not be classified as non-hab
    """

    declared_nonhab = []
    unknown = []
    ### Before using this file, modify Parameters.py to fit desried values ###

    for row in dataset:
        # This takes all instances of ' be ' and the next 25 characters
        # if the next 25 characters have -ing, then it is habitual

        tokenizer = RegexpTokenizer(r"\w+")
        tokenized = tokenizer.tokenize(row[0])
        token_list = pos_tag(tokenized)

        i = row[1].astype(np.int)
        if (token_list[i][0] != 'be'):
            if (token_list[i - 1][0] == 'be'):
                i = i - 1
            elif(token_list[i-2][0] == 'be'):
                i = i-2
            elif (token_list[i + 2][0] == 'be'):
                i = i + 2
            elif(token_list[i+1][0] == 'be'):
                i = i+1



        if (token_list[i][0] != 'be'):
            continue


        if token_list[i - 1][1] == "MD" or token_list[i - 1][1] == "TO" or token_list[i - 1][1] == "JJ":
            declared_nonhab.append([row[0], i, row[2]])


        # Good for 0s
        elif token_list[i + 1][1] == "VBN" and token_list[i - 1][1] != "PRP" and token_list[i - 1][1] != "NN":
            declared_nonhab.append([row[0], i, row[2]])

        # This returned a majority 0, this goes to advective
        elif (token_list[i + 1][1] == "JJ" and token_list[i - 1][1] != "PRP" and token_list[i - 1][1] != "NN" and
             token_list[i - 1][1] != "NNS" and token_list[i - 1][1] != "RB"):
            declared_nonhab.append([row[0], i, row[2]])

        # All 0s, prepositional phrases correspond to IN
        elif token_list[i + 1][1] == "IN" and (
                token_list[i - 1][1] == "TO" or token_list[i - 1][1] == "MD" or token_list[i - 1][1] == "VBP" or
                token_list[i - 1][1] == "VBZ"):
            declared_nonhab.append([row[0], i, row[2]])

        elif (token_list[i - 1][1] == "RB" and (
                token_list[i + 1][1] == "PRP" or token_list[i + 1][1] == "DT" or token_list[i - 2][
            1] == "VBD" or token_list[i - 2][1] == "MD")):
            declared_nonhab.append([row[0], i, row[2]])

        elif (token_list[i - 1][1] == "NN" and token_list[i - 2][1] == "JJ"):
            declared_nonhab.append([row[0], i, row[2]])


        else:
            unknown.append([row[0], i, row[2]])


    return np.array(declared_nonhab), np.array(unknown)

