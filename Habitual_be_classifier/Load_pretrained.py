import pickle

def get_pretrained(filepath = 'Classifiers.obj'):
    with open(filepath, 'rb') as handle:
        classifiers = pickle.load(handle)

    return classifiers