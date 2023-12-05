import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from financial_app.utils import drop_rows, \
                  drop_columns, \
                  scale_dataframe

def preprocessor(df):
    """

    """

    ## Remove useless columns
    df = drop_columns(df)

    ## Drop rows with nan values
    df = drop_rows(df)

    ## Scaling
    df = scale_dataframe(df)

    return df
