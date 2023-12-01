import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from utils import data_formating, \
                sent_df_formating, \
                moving_averages, \
                bollinger_bands, \
                rdp, \
                bias, \
                rsi, \
                ema, \
                macd, \
                psy, \
                williams_percent_r, \
                stochastic_oscillator, \
                labeling_df, \
                merge_df



def get_data():
    data = pd.read_csv("raw_data/cleaned_merged_data-sample-with-sentiment-v02.csv")
    columns = ['Datetime', 'score_int', 'total_tweets', 'share_of_positive', 'share_of_negative']
    sentimental_data = data[columns]

    df = data.copy()
    sent_df = sentimental_data.copy()
    return df, sent_df



def formatting_merging(df, sent_df):

    ## Data formating for financial columns
    formated_df = data_formating(df, 'Datetime', 'Open-AAPL', 'High-AAPL', 'Low-AAPL', 'Adj Close-AAPL', 'Volume-AAPL')

    ## Data formating for sentiment analysis
    formated_sent_df = sent_df_formating(sent_df, 'Datetime', 'score_int', 'total_tweets', 'share_of_positive', 'share_of_negative')

    ## Labeling with Y
    labeled_formated_df = labeling_df(formated_df)

    ## Merging both df
    merged_df = merge_df(labeled_formated_df, formated_sent_df)

    return merged_df


##df = cleaning_formatting_merging(df)

###############################
## FEATURES ENGINEERING SECTION
###############################


def features_engineering(df):
    df = moving_averages(df)
    df = bollinger_bands(df)
    df = rdp(df)
    df = bias(df)
    df = rsi(df)
    df = ema(df)
    df = macd(df)
    df = psy(df)
    df = williams_percent_r(df)
    df = stochastic_oscillator(df)

    return df
