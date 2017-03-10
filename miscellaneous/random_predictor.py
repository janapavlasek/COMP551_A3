#!/usr/bin/env python
import numpy as np


def count_classes(y_train):
    probs = [0.0] * 40
    for i in range(0, len(y_train)):
        probs[y_train[i]] += 1.0

    for i in range(0, len(probs)):
        probs[i] = probs[i] / float(len(y_train))

    return probs


def score(y_pred, y_expect):
    errors = 0.0
    for i in range(0, len(y_pred)):
        if y_pred[i] != y_expect[i]:
            errors += 1.0

    return 1 - errors / float(len(y_pred))


if __name__ == '__main__':
    TEST_PERCENTAGE = 0.2
    y = np.load("data/tinyY.npy")

    np.random.shuffle(y)

    test_length = int(len(y) * TEST_PERCENTAGE)

    y_test = y[len(y) - test_length - 1:len(y)]
    y_train = y[0:len(y) - test_length - 1]

    probs = count_classes(y_train)

    y_pred = np.empty(y_test.shape)

    for i in range(0, y_pred.shape[0]):
        y_pred[i] = np.random.choice(np.arange(40), p=probs)

    print score(y_pred, y_test)
