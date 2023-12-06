from tensorflow import keras
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import Sequential, layers
from registry import *


def lstm_model_initialization(X_train, window_size=10):

    #############################
    #  1 - Model architecture   #
    #############################
    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = Sequential()
    model.add(normalizer)
    model.add(layers.LSTM(units=10, activation='tanh', return_sequences=True, input_shape=(window_size, X_train.shape[-1]), kernel_regularizer=l2(0.5)))
    model.add(layers.LSTM(units=3, activation='tanh', return_sequences=False))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(3, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))

    #############################
    #  2 - Optimization Method  #
    #############################
    model.compile(loss= 'binary_crossentropy',
                  optimizer = Adam(learning_rate=0.0001),
                  metrics = ['accuracy'])

    return model
