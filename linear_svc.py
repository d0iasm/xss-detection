import argparse
import numpy as np
import pandas as pd
from sklearn import svm, metrics, preprocessing
from sklearn.model_selection import train_test_split


is_test = False

def read():
    with open() as f:
            print(type(f))


def parse_args():
    global is_test

    parser = argparse.ArgumentParser(
            description='Detect XSS based on linear support vector machine classification.')
    parser.add_argument('-test', action='store_true')

    args = parser.parse_args()
    is_test = args.test


if __name__ == '__main__':
    parse_args()
