import pandas as pd
import numpy as np
import re

def findOccurrences(s, ch):
    return [i for i in range(len(s)) if s.startswith(ch, i)]
    #return [i for i, letter in enumerate(s) if letter == ch]

def csv_processor(filePath):

    """
    Creates lists of nonhab and hab "be" instances from a tagged csv
    Parameters
        - file path to a csv. The text input is in a column named 'Concordance' and the
        habituality is indicated by 1/0 (hab/nonhab) in a column titled 'Habituality'
    Returns
        - Two lists. One is of the non-hab instances, the other is of the hab instances
    """

    hab_lst = []

    nonhab_lst = []
    for path in filePath:
        text = pd.read_csv(path, usecols=['Habituality','Concordance'])

        for index, row in text.iterrows():
            input_row = str(text.iloc[index,1]).lower()
            be_indices = findOccurrences(input_row, 'be')
            spaces = findOccurrences(input_row, ' ')


            #assert 48 in spaces or 49 in spaces, 'location of be not where expected'

            diff = np.absolute(np.array(be_indices) - 50)
            be_index = diff.argmin()
            #be_index = spaces.index(48) if 48 in spaces else spaces.index(49)

            if len(spaces) - be_index > 5:
                input_row = " " + input_row + " "
                input_row = re.sub("[^\w\s']", "", input_row) # remove punctuation
                input_row = input_row.replace(" be'", " be ")
                hab = int(text.iloc[index, 0])

                be_indices = findOccurrences(input_row, 'be')
                diff = np.absolute(be_indices - 50)
                be_index = diff.argmin()

                word_index = input_row.count(' ', 0, be_index)
                if hab == 1:
                    hab_lst.append([input_row, word_index+1, 1])
                if hab == 0:
                    nonhab_lst.append([input_row, word_index+1, 0])



    return hab_lst, nonhab_lst




