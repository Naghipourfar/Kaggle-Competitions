import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from sklearn.preprocessing import normalize
"""
    Created by Mohsen Naghipourfar on 4/2/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""
# Constants
TRAINING_PATH = './train.csv'
TEST_PATH = './test.csv'

# Hyper-Parameters
n_features = 81
n_epochs = 1000
batch_size = 256


def load_data(filepath):
    return pd.read_csv(filepath)


def MLP_model():
    model = Sequential()
    # model.add(Input(shape=(n_features,)))
    model.add(Dense(64, input_shape=(n_features,)))
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dropout(0.5))
    model.add(Dense(16))
    model.add(Dropout(0.5))
    model.add(Dense(8))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    model.summary()

    return model


def train_model(model, x_train, y_train):
    model.fit(x=x_train,
              y=y_train,
              epochs=n_epochs,
              batch_size=batch_size,
              validation_split=0.2)


def predict(model, x_test):
    return pd.DataFrame(model.predict(x_test))


def build_submission(x_test, y_pred):
    idx = x_test.columns.get_loc('Id')
    import csv
    with open('./submission.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Id', 'SalePrice'])
        for i in range(len(y_pred)):
            writer.writerow([x_test.iloc[i, idx], y_pred[i][0]])


def preprocess_data(data, test_data):
    # 1.1. Drop Columns which Contains Missing Values in all elements
    # data = data.dropna(axis=1, how='all')
    # 1. Drop All Missing Data is a Fantastic Way
    data, test_data = process_missing_data(data, test_data)
    # 1.2. Drop Rows Containing Any Missing Values (NaN)
    # data = data.dropna(axis=0, how='any')

    # 2. Convert Categorical to Dummies
    n_train_samples = data.shape[0]

    y_train = data['SalePrice']
    train_data = data.drop('SalePrice', axis=1)
    all_data = train_data.append(test_data)
    all_data = pd.get_dummies(all_data)

    x_train = all_data.iloc[:n_train_samples, :]
    x_test = all_data.iloc[n_train_samples:, :]

    print(x_train.shape, x_test.shape)

    # 3. Normalize Data
    # data = normalize(data, axis=1)
    return x_train, y_train, x_test


def process_missing_data(data, test_data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    indices = (missing_data[missing_data['Total'] > 1]).index
    data = data.drop(indices, 1)
    test_data = test_data.drop(indices, 1)

    data = data.drop(data.loc[data['Electrical'].isnull()].index)

    # data.isnull().sum().max()  # just checking that there's no missing data missing...
    return data, test_data


if __name__ == '__main__':
    training_data = load_data(TRAINING_PATH)
    test_data = load_data(TEST_PATH)

    x_train, y_train, x_test = preprocess_data(training_data, test_data)

    # x_train = training_data.drop(labels=['SalePrice', 'Id'], axis=1)
    # y_train = training_data['SalePrice']

    # x_train = pd.DataFrame(normalize(x_train, axis=1))

    n_features = x_train.shape[1] - 1

    model = MLP_model()

    train_model(model, x_train.drop('Id', axis=1).as_matrix(), y_train.as_matrix())

    y_test = predict(model, x_test.drop('Id', axis=1).as_matrix())

    build_submission(x_test, y_test.values.tolist())
