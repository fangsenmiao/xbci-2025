import numpy as np
from npy_folder_loader import loadnpyfolder

root = 'data/'
#total_data = loadnpyfolder(root + 'T')
total_data = loadnpyfolder(root)
data = total_data['data']
labels = total_data['label']

#print("data:", data.shape, data)

import pywt


# signal is decomposed to level 5 with 'db4' wavelet

def wpd(X):
    coeffs = pywt.WaveletPacket(X, 'db4', mode='symmetric', maxlevel=6)
    return coeffs


def feature_bands(x):
    Bands = np.empty((8, x.shape[0], x.shape[1], 38))  # 8 freq band coefficients are chosen from the range 4-32Hz

    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
            pos = []
            C = wpd(x[i, ii, :])
            pos = np.append(pos, [node.path for node in C.get_level(6, 'natural')])
            for b in range(1, 9):
                Bands[b - 1, i, ii, :] = C[pos[b]].data

    return Bands

#print("data:", data.shape, data)
wpd_data = feature_bands(data)

from mne.decoding import CSP # Common Spatial Pattern Filtering
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
from sklearn.model_selection import ShuffleSplit

# OneHotEncoding Labels
enc = OneHotEncoder()
X_out = enc.fit_transform(labels.reshape(-1,1)).toarray()

# Cross Validation Split
cv = ShuffleSplit(n_splits = 10, test_size = 0.2, random_state = 0)

from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

acc = []
ka = []
prec = []
recall = []

def build_classifier(num_layers = 1):
    classifier = Sequential()
    #First Layer
    classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'relu', input_dim = 32,
                         kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
    classifier.add(Dropout(0.5))
    # Intermediate Layers
    for itr in range(num_layers):
        classifier.add(Dense(units = 124, kernel_initializer = 'uniform', activation = 'relu',
                             kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
        classifier.add(Dropout(0.5))
    # Last Layer
    classifier.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier


for train_idx, test_idx in cv.split(labels):
    Csp = [];
    ss = [];
    nn = []  # empty lists

    label_train, label_test = labels[train_idx], labels[test_idx]
    y_train, y_test = X_out[train_idx], X_out[test_idx]

    # CSP filter applied separately for all Frequency band coefficients

    Csp = [CSP(n_components=4, reg=None, log=True, norm_trace=False) for _ in range(8)]
    ss = preprocessing.StandardScaler()

    X_train = ss.fit_transform(
        np.concatenate(tuple(Csp[x].fit_transform(wpd_data[x, train_idx, :, :], label_train) for x in range(8)),
                       axis=-1))

    X_test = ss.transform(
        np.concatenate(tuple(Csp[x].transform(wpd_data[x, test_idx, :, :]) for x in range(8)), axis=-1))

    nn = build_classifier()

    nn.fit(X_train, y_train, batch_size=25, epochs=30)

    y_pred = nn.predict(X_test)
    pred = (y_pred == y_pred.max(axis=1)[:, None]).astype(int)

    acc.append(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    ka.append(cohen_kappa_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
    prec.append(precision_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))
    recall.append(recall_score(y_test.argmax(axis=1), pred.argmax(axis=1), average='weighted'))

import pandas as pd

scores = {'Accuracy': acc, 'Kappa': ka, 'Precision': prec, 'Recall': recall}

Es = pd.DataFrame(scores)

avg = {'Accuracy': [np.mean(acc)], 'Kappa': [np.mean(ka)], 'Precision': [np.mean(prec)],
           'Recall': [np.mean(recall)]}

Avg = pd.DataFrame(avg)

T = pd.concat([Es, Avg])

T.index = ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'Avg']
T.index.rename('Fold', inplace=True)

print(T)