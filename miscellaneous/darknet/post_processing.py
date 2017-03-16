#!/usr/bin/env python
import numpy as np
from random import shuffle


def count_classes(y_train):
    probs = [0.0] * 40
    for i in range(0, len(y_train)):
        probs[y_train[i]] += 1.0

    for i in range(0, len(probs)):
        probs[i] = probs[i] / float(len(y_train))

    return probs


if __name__ == '__main__':
    y = np.load("data/tinyY.npy")

    with open("results/darknet19_train_predictions.txt", "r") as f:
        X = f.readlines()

    with open("results/darknet19_test_predictions.txt", "r") as f:
        X_test = f.readlines()

    ids = [i for i in range(0, len(X))]

    probs = count_classes(y)

    data = zip(ids, X)
    shuffle(data)

    # data = data[0:10]

    split = int(len(X) * 0.8)

    data_test = data[0:split]
    data_train = data[split + 1:len(X) - 1]

    out = []
    sum_valid = 0.0
    not_found_valid = 0

    # Cross validation.
    for ele in data_test:
        result = []
        for ans in data_train:
            if ele[1] == ans[1]:
                result.append(y[ans[0]])

        if len(result) == 0:
            not_found_valid += 1
            result = [np.random.choice(np.arange(40), p=probs)]

        guess = max(set(result), key=result.count)

        if guess == y[ele[0]]:
            sum_valid += 1.0

    classes = []
    not_found_test = 0
    for ele in X_test:
        results = []
        for ans in data:
            if ele == ans[1]:
                results.append(y[ans[0]])

        if len(results) == 0:
            not_found_test += 1
            results = [np.random.choice(np.arange(40), p=probs)]

        guess = max(set(results), key=results.count)

        classes.append(guess)

    with open("results/darknet19_predictions.csv", "w") as f:
        f.write("id,class\n")
        for i, cl in enumerate(classes):
            f.write("{},{}\n".format(i, cl))

    print "Accuracy:", sum_valid / float(len(data_test))
    print "Randomly classified", not_found_valid, "/", len(data_test)
    print "Randomly classified test", not_found_test, "/", len(X_test)
