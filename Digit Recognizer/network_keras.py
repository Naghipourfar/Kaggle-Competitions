import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras.layers import Dense, Input, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

"""
    Created by Mohsen Naghipourfar on 3/30/18.
    Email : mn7697np@gmail.com
    Website: http://ce.sharif.edu/~naghipourfar
"""
# Hyper-Parameters
n_epochs = 10
n_batch_size = 512


def MLP(x_train, y_train):
    input_image = Input(shape=(784,))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_image)

    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)


def CNN(x_train, y_train):
    network = design_network()

    network.summary()

    checkpoint = ModelCheckpoint(filepath='./checkpoint.hdf5',
                                 verbose=0,
                                 monitor='val_acc',
                                 save_best_only=True,
                                 mode='auto',
                                 period=1)

    network.fit(x=x_train,
                y=y_train,
                epochs=n_epochs,
                batch_size=n_batch_size,
                validation_split=0.2,
                callbacks=[checkpoint])

    x_test = pd.read_csv('./test.csv').as_matrix()
    x_test = x_test.reshape(28000, 28, 28, 1)
    prediction = network.predict(x=x_test)

    network.save('./model.hdf5')
    return prediction


def design_network():
    network = Sequential()
    network.add(Conv2D(32, (3, 3), activation='relu', padding='valid', input_shape=(28, 28, 1)))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Conv2D(64, (3, 3), activation='relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Conv2D(128, (3, 3), activation='relu'))
    network.add(MaxPooling2D(pool_size=(2, 2)))
    network.add(Flatten())
    network.add(Dense(512, activation='relu'))
    network.add(Dropout(0.5))
    network.add(Dense(10, activation='softmax'))

    network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return network


def load_data():
    training_data = pd.read_csv('./train.csv')
    return training_data.drop('label', 1), training_data['label']


def load_pretrained_model(x_train, y_train):
    network = keras.models.load_model('./model.hdf5')
    network.load_weights('./checkpoint.hdf5')

    checkpoint = ModelCheckpoint(filepath='./checkpoint.hdf5',
                                 verbose=0,
                                 monitor='val_acc',
                                 save_best_only=True,
                                 mode='auto',
                                 period=1)

    network.fit(x=x_train,
                y=y_train,
                epochs=n_epochs,
                batch_size=n_batch_size,
                validation_split=0.2,
                callbacks=[checkpoint])

    x_test = pd.read_csv('./test.csv').as_matrix()
    x_test = x_test.reshape(28000, 28, 28, 1)
    prediction = network.predict(x=x_test)

    network.save('./model.hdf5')
    return prediction


def build_submission(prediction):
    import csv
    with open('./submission_keras.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['ImageId', 'Label'])
        for i in range(prediction.shape[0]):
            writer.writerow([i + 1, np.argmax(prediction[i])])


if __name__ == '__main__':
    x_train, y_train = load_data()
    x_train = x_train.as_matrix()
    x_train = x_train.reshape(42000, 28, 28, 1)
    y_train = y_train.as_matrix()
    y_train = keras.utils.to_categorical(y_train, 10)
    prediction = load_pretrained_model(x_train, y_train)
    build_submission(prediction)
