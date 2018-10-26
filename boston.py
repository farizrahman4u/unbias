from os.path import join, dirname, abspath
import numpy as np


def get_data(split=0.9):
    boston_csv = join(dirname(abspath(__file__)), 'boston.csv')

    X = []
    Y = []

    with open(boston_csv, 'r') as f:
        _ = f.readline()
        rows = f.readlines()
        np.random.shuffle(rows)

    for row in rows:
        cols = row[:-1].split(',')
        if len(cols) == 14:
            x = list(map(float, cols[:-1]))
            y = float(cols[-1])
            X.append(x)
            Y.append(y)

    X = np.array(X)
    Y = np.array(Y)

    if split is None:
        return X, Y

    num_samples = len(X)
    num_train = int(split * num_samples)

    X_train = X[:num_train]
    Y_train = Y[:num_train]

    X_test = X[num_train:]
    Y_test = Y[num_train:]

    return (X_train, Y_train), (X_test, Y_test)
