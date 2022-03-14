import random
import numpy as np
import nlpaug.augmenter.word as naw
import os
import nltk
from nlpaug.util.file.download import DownloadUtil



#DownloadUtil.download_word2vec(dest_dir='.') # Download word2vec model
#DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.') # Download GloVe model
#DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.') # Download fasttext model

def findOccurrences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

def augmenter(dataset, filepath = '.'):
    nltk.download('wordnet')

    if not os.path.exists(filepath + '/glove.6B.100d.txt'):
        DownloadUtil.download_glove(model_name='glove.6B', dest_dir=filepath ) # Download GloVe model
    if not os.path.exists(filepath + '/GoogleNews-vectors-negative300.bin'):
        DownloadUtil.download_word2vec(dest_dir=filepath) # Download word2vec model
        src = filepath + '/GoogleNews-vectors-negative300.zip'
        dst = filepath + '/GoogleNews-vectors-negative300.bin'
        command = 'gunzip -c -S zip ' + src + ' > ' + dst
        os.system(command)

    if not os.path.exists(filepath + '/wiki-news-300d-1M.vec'):
        DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir=filepath) # Download fasttext model


    hab_filter = dataset[:,2].astype(np.int) == 1
    hab_dataset = dataset[hab_filter]

    num_hab = hab_dataset.shape[0]
    num_nonhab = dataset.shape[0] - num_hab
    times_to_aug = num_nonhab - int(num_hab * 0.4)

    to_ret = []

    for i in range(times_to_aug):
        row_to_aug = random.randint(0, num_hab-1)
        method = random.randint(0, 3)

        row = hab_dataset[row_to_aug]
        index_of_be = row[1].astype(np.int)
        row_text = row[0]
        row_before = row_text[:index_of_be - 1]
        row_after = row_text[index_of_be + 3:]

        if method == 0:
            aug = naw.SynonymAug(aug_src='wordnet')

        elif method == 1:
            aug = naw.WordEmbsAug(
                model_type='word2vec', model_path=filepath + '/GoogleNews-vectors-negative300.bin',
                action="substitute")

        elif method == 2:
            aug = naw.WordEmbsAug(
                model_type='word2vec', model_path=filepath + '/GoogleNews-vectors-negative300.bin',
                action="insert")

        elif method == 3:
            aug = naw.WordEmbsAug(
                model_type='word2vec', model_path=filepath + '/GoogleNews-vectors-negative300.bin',
                action="substitute")

        aug_row_before = aug.augment(row_before)
        aug_row_after = aug.augment(row_after)
        augmented_row = aug_row_before + " be " + aug_row_after

        new_index = len(findOccurrences(aug_row_before, ' ')) + 1

        to_ret.append([augmented_row, new_index, 1])

    return np.array(to_ret)