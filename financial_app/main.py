import pandas as pd
import numpy as np

from data import get_data, features_basic_formating, sent_and_features_basic_formating, features_engineering
from preprocessor import preprocessor
from training import train_test_split_and_reshape, train_model
from estimators.baseline import baseline_model
from estimators.lstm import lstm_model_initialization
from estimators.gru import gru_model_initialization
from registry import *

## PARAMETERS

test_size = 0.2
window_size=10

time_horizon = '5mn'
model_type = 'lstm'
asset = 'btc'

include_sent = False

columns = ['date', 'open', 'high', 'low', 'adj_close', 'volume']
columns_price = ['date','open', 'adj_close']
columns_sent = ['Datetime', 'score_int', 'total_tweets', 'share_of_positive', 'share_of_negative']

model_name = f'{asset}_{model_type}_{time_horizon}'

###### END PARAMETERS


## Get data locally
if include_sent is True:
    df, sent_df = get_data(include_sent=False)
df = get_data(include_sent=False)
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
if model_type == 'baseline':
    estimator = baseline_model(X_train, window_size, optimizer_name='adam')
elif model_type == 'lstm':
    estimator = lstm_model_initialization(X_train, window_size)
elif model_type == 'gru':
    estimator = gru_model_initialization(X_train, window_size)


## Train the model
model = train_model(model_name, estimator, X_train, y_train, validation_split=0.2, batch_size=64 )
