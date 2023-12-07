from tensorflow import keras
from keras.layers import *
from keras.models import Sequential
from keras import Sequential, layers
from keras.layers import Dense, SimpleRNN, Flatten, LSTM, Bidirectional
from registry import *


def baseline_model(X_train, window_size=10, optimizer_name='adam'):

    #############################
    #  1 - Model architecture   #
    #############################

    # [1, 5] layers
    # nodes per layer [30, 70]
    # activation function: relu
    # optimizer: Adam
    # learning rate = [0.001, 0.1]

    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=(window_size, X_train.shape[-1])))
    model.add(Dropout(0.25))
    model.add(Dense(1, activation='sigmoid'))


    # $CHALLENGIFY_END

    #############################
    #  2 - Optimization Method  #
    #############################
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer_name,
                  metrics=['accuracy'])

    return model
