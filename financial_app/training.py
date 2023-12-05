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

    history = model.fit(X_train, y_train,
                        validation_split=validation_split,
                        batch_size=batch_size,
                        epochs=100,
                        callbacks=[es],
                        verbose=1)

    val_accuracy = np.min(history.history['val_accuracy'])

    params = dict(
        context="train",
        row_count=len(X_train),
    )

    # Save results on the hard drive using financial_app.registry
    save_results(params=params,
                 metrics=dict(accuracy=val_accuracy)
                  )

    # Save model weight on the hard drive (and optionally on GCS too!)
    #breakpoint()
    save_model(model=model)

    # The latest model should be moved to staging

    if MODEL_TARGET == 'mlflow':
        mlflow_transition_model(current_stage="None", new_stage="Staging")

    print("✅ train() done \n")


    return model
