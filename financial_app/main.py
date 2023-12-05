import pandas as pd
import numpy as np

from data import get_data, features_basic_formating, sent_and_features_basic_formating, features_engineering
from preprocessor import preprocessor
from training import train_test_split_and_reshape, train_model
from estimators.baseline import baseline_model
from registry import *

## PARAMETERS

test_size = 0.2
window_size=15

include_sent = False

columns = ['Datetime', 'Open-TSLA', 'High-TSLA', 'Low-TSLA', 'Adj Close-TSLA', 'Volume-TSLA']
columns_price = ['Datetime','Open-TSLA', 'Adj Close-TSLA']
columns_sent = ['Datetime', 'score_int', 'total_tweets', 'share_of_positive', 'share_of_negative']

selected_model = 'baseline'

###### END PARAMETERS


## Get data locally
df, sent_df = get_data(columns_sent)

## Formatting and merging data
if include_sent is True:
    df = sent_and_features_basic_formating(df, sent_df, columns_sent, columns)

else:
    df = features_basic_formating(df, columns)


## Enrich the df with features engineering
df = features_engineering(df)

## Preprocessing step
preprocessed_df = preprocessor(df)

## Train split step
X_train, y_train, X_test, y_test = train_test_split_and_reshape(preprocessed_df, test_size, window_size)


## Instantiate the model -- SELECT THE MODEL YOU WANT -- WIP
if selected_model == 'baseline':
    estimator = baseline_model(X_train, window_size, optimizer_name='adam')

## Train the model
model = train_model(estimator, X_train, y_train, validation_split=0.2, batch_size=64 )
