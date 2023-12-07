from tensorflow import keras
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2, l1_l2
from keras import Sequential, layers
from registry import *


def gru_model_initialization(X_train, window_size=10):

    #############################
    #  1 - Model architecture   #
    #############################
    normalizer = Normalization()
    normalizer.adapt(X_train)

    model = Sequential()
    model.add(GRU(16, return_sequences=False, input_shape=(window_size, X_train.shape[-1]), kernel_regularizer=l1_l2(l1=0, l2=0.5)))
    model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(Dense(8, activation='relu'))
    # model.add(GRU(8, return_sequences=False, activation='tanh'))
    # model.add(Dense(8, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    # model.add(GRU(8, return_sequences=False, activation='tanh'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.2))
    # model.add(GRU(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    #############################
    #  2 - Optimization Method  #
    #############################
    model.compile(loss= 'binary_crossentropy',
                  optimizer = Adam(learning_rate=0.0001),
                  metrics = ['accuracy'])

    return model
