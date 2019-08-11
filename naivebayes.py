import argparse
import collections
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import KFold


test_code = ""
test_result = ""


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
    return t


def vectorize(tokens):
    # TODO: Vectorize based on TF-IDF
    d = collections.defaultdict(int)
    for k in tokens:
        d[k] += 1
    return [n for _, n in d.items()]


def cross_validation(model):
    # 4-fold
    kf = KFold(n_splits=4)
    for train_idx, test_idx in kf.split(training, target):
        X_train = training[train_idx[0]:train_idx[len(train_idx)-1]]
        X_test = training[test_idx[0]:test_idx[len(test_idx)-1]]
        y_train = target[train_idx[0]:train_idx[len(train_idx)-1]]
        y_test = target[test_idx[0]:test_idx[len(test_idx)-1]]
        print(model.score(X_test, y_test))


def result(y):
    # XSS is 1 and normal is 0.
    return 1 if y.sum() >= len(y) else 0


def parse_args():
    global test_code
    global test_result

    parser = argparse.ArgumentParser(
            description='Detect XSS based on linear support vector machine classification.')
    parser.add_argument('-test', '--test', nargs=2)

    args = parser.parse_args()
    test_code = args.test[0]
    test_result = args.test[1]


if __name__ == '__main__':
    parse_args()
    training = []
    target = []

    xss_data = read('./dataset/level1/xss')
    xss_vec = vectorize(xss_data)
    normal_data = read('./dataset/level1/normal')
    normal_vec = vectorize(normal_data)

    training += xss_vec
    training += normal_vec
    training = np.array(training).reshape(-1, 1)

    target += [1 for _ in xss_vec] # XSS is 1
    target += [0 for _ in normal_vec] # Normal is 0
    target = np.array(target)

    model = GaussianNB()
    model.fit(training, target)
    print(model.score(training, target))

    cross_validation(model)

    if (test_code):
        print("Validate:", test_code)
        vec = np.array(vectorize(test_code)).reshape(-1, 1)
        a = result(model.predict(vec))
        print("Predict:", a, "XSS" if a == 1 else "Normal")
        print("Expect:", test_result)


