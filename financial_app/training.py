import pandas as pd
import numpy as np
from tensorflow import keras
from keras.layers import *
from keras.models import Sequential
from keras import Sequential, layers
from keras.layers import Dense, SimpleRNN, Flatten, LSTM, Bidirectional
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras import models, layers
from registry import *

from utils import train_test_split, \
                  input_matrix_split_X_y

def train_test_split_and_reshape(df, test_size, window_size:int):

    ## Test split
    split_scaled_clean_merged_df = train_test_split(df, test_size)

    ## Get X_train, y_train, X_test, y_test reshaped
    X_train, y_train = input_matrix_split_X_y(split_scaled_clean_merged_df[0], window_size)
    X_test, y_test = input_matrix_split_X_y(split_scaled_clean_merged_df[1], window_size)

    return X_train, y_train, X_test, y_test

@mlflow_run
def train_model(model, X_train, y_train, validation_split=0.2, batch_size=64 ):

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(X_train, y_train,
                        validation_split=validation_split,
                        batch_size=batch_size,
                        epochs=100,
                        callbacks=[es],
                        verbose=1)

    return model
