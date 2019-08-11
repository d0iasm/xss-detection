import argparse
import collections
import re
import logging
import numpy as np
import pandas as pd
#from gensim.models import Word2Vec
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE


is_test = False
src = ""

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def read(src):
    data = []
    with open(src) as f:
        for line in f:
            data += clean(line).split()
    return data


def clean(t):
    t = t.replace("</", " ")
    t = t.replace("<", " ")
    t = t.replace(">", " ")
    t = t.replace("=", " ")
    t = t.replace("\'", "")
    t = t.replace("\"", "")
    #return t.encode('utf-8')
    return t


def vectorize(corpus):
    return collections.Counter(corpus)
    #return Word2Vec(corpus)


def parse_args():
    global is_test
    global src

    parser = argparse.ArgumentParser(
            description='Detect XSS based on linear support vector machine classification.')
    parser.add_argument('-src', required=True)
    parser.add_argument('-test', action='store_true')

    args = parser.parse_args()
    is_test = args.test
    src = args.src


if __name__ == '__main__':
    parse_args()
    text = read(src)
    vec = vectorize(text)
    print(vec)

    digits = datasets.load_digits()
    print(digits.data)

    #model = word2vec(text)
    #print()
    #print(model.wv.vocab)
    #print()
    #print(model.wv.index2word)
    #print()
    #print(model.wv.syn0)
    #print(model.wv.syn0.shape)

    #print(data)
    #digits = datasets.load_digits()
    #print(digits.data)
    # (1797, 64)
    #print(digits.target.shape)
    # (1797,)
    #X_reduced = TSNE(n_components=2, random_state=0).fit_transform(digits.data)
    #print(digits.data)
    #print(X_reduced.shape)
    # (1797, 2)
    #plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=digits.target)
    #plt.colorbar()
    #plt.show()
