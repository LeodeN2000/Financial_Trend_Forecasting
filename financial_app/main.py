import pandas as pd
import numpy as np

from data import get_data, formatting_merging, features_engineering
from preprocessor import preprocessor
from training import train_test_split_and_reshape, train_model
from estimators.baseline import baseline_model
from registry import *

## PARAMETERS

test_size = 0.2
window_size=5


## Get data locally
df, sent_df = get_data()

## Formatting and merging data
merged_df = formatting_merging(df, sent_df)

## Enrich the df with features engineering
df = features_engineering(merged_df)

## Preprocessing step
preprocessed_df = preprocessor(df)

## Train split step
X_train, y_train, X_test, y_test = train_test_split_and_reshape(df, test_size, window_size)

## Instantiate the model -- SELECT THE MODEL YOU WANT
estimator = baseline_model(X_train, window_size=5, optimizer_name='adam')

## Train the model
model = train_model(estimator, X_train, y_train, validation_split=0.2, batch_size=64 )
