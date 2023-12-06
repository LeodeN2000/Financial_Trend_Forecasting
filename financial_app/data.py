import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from financial_app.utils import data_formating, \
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
                merge_df, \
                proc, \
                momentum, \
                first_order_lag, \
                trading_volume



def get_data(include_sent=False, columns_sent = None):
    data = pd.read_csv("raw_data/pro_btc_60min_price_df_v2.csv")
    if include_sent:
        sentimental_data = data[columns_sent]
        sent_df = sentimental_data.copy()
        df = data.copy()

        return df, sent_df

    df = data.copy()

    return df

# def price_basic_formating(df, columns_price):

#     price_formated_df = price_df_formating(df, columns_price)
#     price_labeled_df = labeling_df(price_formated_df)

#     return price_labeled_df

## CASE 1 FORMATING PRICE DATASETS ONLY
def features_basic_formating(df, columns):

    formated_df = data_formating(df, columns)
    labeled_df = labeling_df(formated_df)

    return labeled_df

## CASE 2 FORMATING AND MERGING PRICE AND SENT DATASETS
def sent_and_features_basic_formating(df, sent_df, columns_sent, columns):

    formated_df = data_formating(df, columns)
    labeled_df = labeling_df(formated_df)
    sent_formated_df = sent_df_formating(sent_df, columns_sent)
    merged_df = merge_df(labeled_df, sent_formated_df)

    return merged_df


# def formatting_merging(df, sent_df):

#     columns_price = ['Datetime','Open-TSLA', 'Adj Close-TSLA']


#     ## Price formating and labelling
#     price_labeled_df = price_basic_formating(df, columns_price)

#     ## Data formating for financial columns
#     #formated_df = data_formating(df, 'Datetime', 'Open-AAPL', 'High-AAPL', 'Low-AAPL', 'Adj Close-AAPL', 'Volume-AAPL')

#     ## Data formating for sentiment analysis
#     #formated_sent_df = sent_df_formating(sent_df, 'Datetime', 'score_int', 'total_tweets', 'share_of_positive', 'share_of_negative')

#     ## Labeling with Y
#     #labeled_formated_df = labeling_df(formated_df)

#     ## Merging both df
#     merged_df = merge_df(labeled_formated_df, formated_sent_df)

#     return merged_df


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
    df = proc(df)
    df = momentum(df)
    df = first_order_lag(df)
    df = trading_volume(df)

    return df
