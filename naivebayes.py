import argparse
import collections
import math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, ComplementNB
from sklearn.model_selection import KFold


keys = []
test_src = ""
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
    total = sum([len(t) for t in tokens])
    vec = []

    # Count term frequency.
    dicts = []
    for t in tokens:
        d = collections.defaultdict(int)
        for k in t:
            d[k] += 1
        dicts.append(d)

    # Calculate TF-IDF.
    for i, t in enumerate(tokens):
        for term in t:
            # TF(ti, dj) = (The count ti in dj) / (The total count ti in all documents)
            tf = dicts[i][term] / sum([d[term] for d in dicts])

            # IDF(ti) = log(The number of documents / The number of documents that has ti)
            idf = math.log10(total / sum([1 if d[term] > 0 else 0 for d in dicts]) + 1)
            vec.append([term, tf * idf])

    return vec


def apply_vec(tokens, vec):
    new_vec = [[i, 0] for i, _ in vec]
    print()
    print("APPLY VEC:", len(vec), len(new_vec), new_vec)
    print()
    keys = {x[0]: idx for idx, x in enumerate(new_vec)}
    print('----------------------')
    print("KEYS:", keys)
    print('----------------------')
    for t in tokens:
        if t in keys:
            new_vec[keys[t]][1] += 1
    print()
    print("APPLY VEC:", len(vec), len(new_vec), new_vec)
    print()
    return new_vec


def cross_validation(model):
    # 4-fold
    kf = KFold(n_splits=4)
    for train_idx, test_idx in kf.split(training, target):
        X_train = training[train_idx[0]:train_idx[len(train_idx)-1]]
        X_test = training[test_idx[0]:test_idx[len(test_idx)-1]]
        y_train = target[train_idx[0]:train_idx[len(train_idx)-1]]
        y_test = target[test_idx[0]:test_idx[len(test_idx)-1]]
        print(model.score(X_test, y_test))


def result(y, threshold):
    # XSS is 1 and normal is 0.
    print('RESULT: prediction sum(), length of prediction, threshold')
    print(y.sum())
    print(len(y))
    print(threshold)
    print('-----------------------')
    return 1 if y.sum()/len(y) >= threshold else 0


def parse_args():
    global test_src
    global test_result

    parser = argparse.ArgumentParser(
            description='Detect XSS based on linear support vector machine classification.')
    parser.add_argument('-test', '--test', nargs=2)

    args = parser.parse_args()
    if (args.test):
        test_src = args.test[0]
        test_result = args.test[1]


if __name__ == '__main__':
    parse_args()

    normal_data = read('./dataset/level1/normal')
    xss_data = read('./dataset/level1/xss')
    training_data = [normal_data, xss_data]
    vec = vectorize(training_data)

    threshold = len(xss_data) / (len(xss_data) + len(normal_data))

    training = [n for _, n in vec]
    print('-------')
    print("Training:", len(training), training)
    print("  vector: ", vec)
    print('-------')
    training = np.array(training).reshape(-1, 1)

    target = [0 for _ in range(len(normal_data))] # Normal is 0.
    target += [1 for _ in range(len(xss_data))] # XSS is 1.
    print('-------')
    print("Target:", len(target), target)
    print('-------')
    target = np.array(target)

    model = GaussianNB()
    model.fit(training, target)
    # Test with the training data.
    print(model.score(training, target))

    cross_validation(model)

    if (test_src):
        print("Validate:", test_src)
        test_data = read(test_src)
        test_vec = vectorize(training_data + [test_data])
        # test_vec = apply_vec(data, (xss_vec + normal_vec))
        test_vec = np.array([n for _, n in test_vec], dtype=object)
        print("Vector:", test_vec)
        test_vec = test_vec.reshape(-1, 1)
        result = result(model.predict(test_vec), threshold)
        print('==========')
        print("Prediction:", model.predict(test_vec))
        print('==========')
        print("Predict:", result, "XSS" if result == 1 else "Normal")
        print("Expect: ", test_result, "XSS" if test_result == "1" else "Normal")

