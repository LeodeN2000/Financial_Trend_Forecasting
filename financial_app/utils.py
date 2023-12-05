import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np



def drop_columns(df, columns_to_drop=['open', 'high', 'low', 'adj_close', 'volume']):
    """
    Drop specified columns from a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns_to_drop (list): List of column names to drop. Default is ['Open', 'High', 'Low', 'Adj_Close', 'Volume'].

    Returns:
    - pd.DataFrame: DataFrame with specified columns dropped.
    """
    # Drop specified columns
    df = df.drop(columns=columns_to_drop, errors='ignore')

    return df



def drop_rows(df):
    """
    Drop all rows with NaN values from a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with NaN rows dropped.
    - list: List of indices corresponding to dropped rows.
    """
    # Drop rows with NaN values
    cleaned_df = df.dropna()

    return cleaned_df


def data_formating(df, columns):
    """
    Preprocess a DataFrame by renaming columns, setting columns to float64,
    dropping unnecessary columns, setting the 'date' column to datetime type,
    and setting the 'date' column as the index.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns (list): define which columns of df refere to which variable

    Returns:
    - pd.DataFrame: formated DataFrame.
    """
    # Step 1: Rename columns
    formated_df = df.rename(columns={
        columns[0]: 'date',
        columns[1]: 'open',
        columns[2]: 'high',
        columns[3]: 'low',
        columns[4]: 'adj_close',
        columns[5]: 'volume'
    })

    # Step 2: Set columns to float64
    formated_df = formated_df.astype({'open': 'float32', 'high': 'float32', 'low': 'float32', 'adj_close': 'float32', 'volume': 'float32'})

    # Step 3: Drop all other columns
    columns_to_keep = ['date', 'open', 'high', 'low', 'adj_close', 'volume']
    formated_df = formated_df[columns_to_keep]

    # Step 4: Set 'date' column to datetime type
    formated_df['date'] = pd.to_datetime(formated_df['date'], format='%Y-%m-%d %H:%M:%S')

    # Step 5: Set 'date' column as the index
    formated_df.set_index('date', inplace=True)

    return formated_df


def price_df_formating(df, columns_price):
    """
    Preprocess a DataFrame by renaming columns, setting columns to float32,
    dropping unnecessary columns, setting the 'date' column to datetime type,
    and setting the 'date' column as the index.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - columns_price (list): define which columns of df refere to which variable

    Returns:
    - pd.DataFrame: formated DataFrame.
    """
    # Step 1: Rename columns
    formated_df = df.rename(columns={
        columns_price[0]: 'date',
        columns_price[1]: 'open',
        columns_price[2]: 'adj_close'
    })

    # Step 2: Set columns to float64
    formated_df = formated_df.astype({'open': 'float32', 'adj_close': 'float32'})

    # Step 3: Drop all other columns
    columns_to_keep = ['date', 'open', 'adj_close']
    formated_df = formated_df[columns_to_keep]

    # Step 4: Set 'date' column to datetime type
    formated_df['date'] = pd.to_datetime(formated_df['date'], format='%Y-%m-%d %H:%M:%S')

    # Step 5: Set 'date' column as the index
    formated_df.set_index('date', inplace=True)

    # Step 6: Drop Nan rows
    price_formated_df = formated_df.dropna()

    return price_formated_df


def sent_df_formating(sent_df, columns_sent):
    """
    Preprocess a DataFrame by renaming columns, setting columns to float32,
    dropping unnecessary columns, setting the 'date' column to datetime type,
    and setting the 'date' column as the index.

    Parameters:
    - sent_df (pd.DataFrame): Input DataFrame.
    - columns_sent (list): define which columns of df refere to which price data

    Returns:
    - pd.DataFrame: formated DataFrame.
    """
    # Step 1: Rename columns
    sent_df = sent_df.rename(columns={
        columns_sent[0]: 'date',
        columns_sent[1]: 'score',
        columns_sent[2]: 'total',
        columns_sent[3]: 'positive',
        columns_sent[4]: 'negative'
    })

    # Step 2: Set columns to float64
    sent_df = sent_df.astype({'score': 'float32', 'total': 'float32', 'positive': 'float32', 'negative': 'float32'})

    # Step 3: Set 'date' column to datetime type
    sent_df['date'] = pd.to_datetime(sent_df['date'], format='%Y-%m-%d %H:%M:%S')

    # Step 4: Set 'date' column as the index
    sent_df.set_index('date', inplace=True)

    # Step 4: Drop Nan rows
    sent_formated_df = sent_df.dropna()

    return sent_formated_df


def labeling_df(labeled_df):
    # Create a new column 'Label' and initialize with 0 (constant)
    labeled_df['label'] = 0

    # Label -1 (down) where 'Open' is higher than 'Adj Close'
    labeled_df.loc[labeled_df['open'] > labeled_df['adj_close'], 'label'] = 0

    # Label +1 (up) where 'Open' is lower than 'Adj Close'
    labeled_df.loc[labeled_df['open'] < labeled_df['adj_close'], 'label'] = 1

    return labeled_df


def merge_df(df, sent_df):

    # Merge two df on their indexes
    merged_df = pd.merge(df, sent_df, left_index=True, right_index=True)

    return merged_df


def sent_and_features_basic_formating(df, sent_df,  columns_sent, columns):


    formated_df = data_formating(df, columns)
    labeled_df = labeling_df(formated_df)
    sent_formated_df = sent_df_formating(sent_df, columns_sent)
    merged_df = merge_df(labeled_df, sent_formated_df)

    return merged_df



#### FEATURES ENGINEERING

def moving_averages(df, column_name='adj_close', window_sizes=[5, 20]):
    """
    Add Moving Averages (MA) columns to the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which moving averages are calculated.
    - window_sizes (list): List of window sizes for moving averages. Default is [5, 20].

    Returns:
    - pd.DataFrame: DataFrame with added MA columns.
    """
    for window_size in window_sizes:
        ma_column_name = f'MA_{window_size}'
        df[ma_column_name] = df[column_name].rolling(window=window_size).mean()

    return df

### B. Bollinger Band (BB up & BB down)

def bollinger_bands(df, column_name='adj_close', window_size=20, num_std_dev=2):
    """
    Calculate Bollinger Bands for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which Bollinger Bands are calculated.
    - window_size (int): Window size for the moving average. Default is 20.
    - num_std_dev (int): Number of standard deviations for the upper and lower bands. Default is 2.

    Returns:
    - pd.DataFrame: DataFrame with added columns for Bollinger Bands (BB up, BB down).
    """
    # Calculate the rolling mean (middle band)
    df['middle_band'] = df[column_name].rolling(window=window_size).mean()

    # Calculate the rolling standard deviation
    df['stddev'] = df[column_name].rolling(window=window_size).std()

    # Calculate Bollinger Bands
    df['bb_Up'] = df['middle_band'] + num_std_dev * df['stddev']
    df['bb_Down'] = df['middle_band'] - num_std_dev * df['stddev']

    # Drop intermediate columns
    df.drop(['middle_band', 'stddev'], axis=1, inplace=True)

    return df


### C. Relative Difference in the Percentage of the price (RDP(1))

def rdp(df, column_name='adj_close'):
    """
    Calculate Relative Difference in the Percentage of the price (RDP(1)) for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which RDP(1) is calculated.

    Returns:
    - pd.DataFrame: DataFrame with an added column for RDP(1).
    """
    # Calculate RDP(1)
    df['rdp_1'] = df[column_name].pct_change() * 100

    return df

### D. Bias Ratio (BIAS(6), BIAS(12) & BIAS(24))

def bias(df, column_name='adj_close', ma_windows=[6, 12, 24]):
    """
    Calculate Bias Ratios (BIAS) for specified moving average windows for a column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which BIAS is calculated.
    - ma_windows (list): List of moving average window sizes. Default is [6, 12, 24].

    Returns:
    - pd.DataFrame: DataFrame with added columns for BIAS(6), BIAS(12), and BIAS(24).
    """
    for window_size in ma_windows:
        ma_column_name = f'MA_{window_size}'
        bias_column_name = f'BIAS_{window_size}'

        # Calculate the moving average
        df[ma_column_name] = df[column_name].rolling(window=window_size).mean()

        # Calculate BIAS
        df[bias_column_name] = ((df[column_name] - df[ma_column_name]) / df[ma_column_name]) * 100

        # Drop intermediate columns
        df.drop(ma_column_name, axis=1, inplace=True)

    return df


### E. Relative Strength Index (RSI)

def rsi(df, column_name='adj_close', window=14):
    """
    Calculate the Relative Strength Index (RSI) for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which RSI is calculated. Default is 'Close'.
    - window (int): Window size for RSI calculation. Default is 14.

    Returns:
    - pd.DataFrame: DataFrame with an added column for RSI.
    """
    # Calculate daily price changes
    df['price_change'] = df[column_name].diff()

    # Calculate the average gain and average loss over the specified window
    df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0).rolling(window=window, min_periods=1).mean()
    df['loss'] = -df['price_change'].apply(lambda x: x if x < 0 else 0).rolling(window=window, min_periods=1).mean()

    # Calculate relative strength (RS)
    df['rs'] = df['gain'] / df['loss']

    # Calculate RSI
    df['rsi'] = 100 - (100 / (1 + df['rs']))

    # Drop intermediate columns
    df.drop(['price_change', 'gain', 'loss', 'rs'], axis=1, inplace=True)

    return df



### F. Exponential Moving Average (EMA(12) & EMA(26))

def ema(df, column_name='adj_close', ema_short=12, ema_long=26):
    """
    Calculate Exponential Moving Averages (EMA) for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which EMA is calculated. Default is 'Close'.
    - ema_short (int): Short-term EMA window size. Default is 12.
    - ema_long (int): Long-term EMA window size. Default is 26.

    Returns:
    - pd.DataFrame: DataFrame with added columns for EMA(12) and EMA(26).
    """
    # Calculate EMA(12)
    df['ema_12'] = df[column_name].ewm(span=ema_short, adjust=False).mean()

    # Calculate EMA(26)
    df['ema_26'] = df[column_name].ewm(span=ema_long, adjust=False).mean()

    return df

### G. Moving Average Convergence/Divergence (MACD)

def macd(df, column_name='adj_close', ema_short=12, ema_long=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD) and its signal line for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which MACD is calculated. Default is 'Close'.
    - ema_short (int): Short-term EMA window size. Default is 12.
    - ema_long (int): Long-term EMA window size. Default is 26.
    - signal_period (int): Signal line EMA window size. Default is 9.

    Returns:
    - pd.DataFrame: DataFrame with added columns for MACD, Signal Line, and MACD Histogram.
    """
    # Calculate short-term EMA
    df['ema_short'] = df[column_name].ewm(span=ema_short, adjust=False).mean()

    # Calculate long-term EMA
    df['ema_long'] = df[column_name].ewm(span=ema_long, adjust=False).mean()

    # Calculate MACD Line
    df['dif'] = df['ema_short'] - df['ema_long']

    # Calculate Signal Line
    df['signal_line'] = df['dif'].ewm(span=signal_period, adjust=False).mean()

    # Calculate MACD Histogram
    df['osc'] = df['dif'] - df['signal_line']

    # Drop intermediate columns
    df.drop(['ema_short', 'ema_long'], axis=1, inplace=True)

    return df

### H. Psychological Line (PSY(12) & PSY(24))

def psy(df, column_name='adj_close', psy_short=12, psy_long=24):
    """
    Calculate Psychological Line (PSY) for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which PSY is calculated. Default is 'Close'.
    - psy_short (int): Short-term PSY window size. Default is 12.
    - psy_long (int): Long-term PSY window size. Default is 24.

    Returns:
    - pd.DataFrame: DataFrame with added columns for PSY(12) and PSY(24).
    """
    # Calculate the percentage of days where the closing price is higher than the previous day's closing price
    df['price_up'] = df[column_name].diff() > 0

    # Calculate PSY(12)
    df['psy_12'] = df['price_up'].rolling(window=psy_short).mean() * 100

    # Calculate PSY(24)
    df['psy_24'] = df['price_up'].rolling(window=psy_long).mean() * 100

    # Drop intermediate columns
    df.drop(['price_up'], axis=1, inplace=True)

    return df

### I. Williams %R (WMS%R)

def williams_percent_r(df, high_column='high', low_column='low', adj_close_column='adj_close', window=14):
    """
    Calculate Williams %R for a specified high, low, and close columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - high_column (str): Name of the column containing high prices. Default is 'High'.
    - low_column (str): Name of the column containing low prices. Default is 'Low'.
    - adj_close_column (str): Name of the column containing close prices. Default is 'Close'.
    - window (int): Window size for Williams %R calculation. Default is 14.

    Returns:
    - pd.DataFrame: DataFrame with an added column for Williams %R.
    """
    # Calculate highest high and lowest low over the specified window
    df['hh'] = df[high_column].rolling(window=window).max()
    df['ll'] = df[low_column].rolling(window=window).min()

    # Calculate Williams %R
    df['williams_r'] = (df['hh'] - df[adj_close_column]) / (df['hh'] - df['ll']) * -100

    # Drop intermediate columns
    df.drop(['hh', 'll'], axis=1, inplace=True)

    return df

### J. Stochastic Oscillator (Stochastic%K & Stochastic%D)

def stochastic_oscillator(df, high_column='high', low_column='low', adj_close_column='adj_close', k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator (%K and %D) for specified high, low, and close columns in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - high_column (str): Name of the column containing high prices. Default is 'High'.
    - low_column (str): Name of the column containing low prices. Default is 'Low'.
    - close_column (str): Name of the column containing close prices. Default is 'Close'.
    - k_window (int): Window size for %K calculation. Default is 14.
    - d_window (int): Window size for %D calculation. Default is 3.

    Returns:
    - pd.DataFrame: DataFrame with added columns for Stochastic %K and %D.
    """
    # Calculate lowest low and highest high over the specified window
    df['ll'] = df[low_column].rolling(window=k_window).min()
    df['hh'] = df[high_column].rolling(window=k_window).max()

    # Calculate Stochastic %K
    df['stochastic_k'] = ((df[adj_close_column] - df['ll']) / (df['hh'] - df['ll'])) * 100

    # Calculate Stochastic %D (3-day simple moving average of %K)
    df['stochastic_d'] = df['stochastic_k'].rolling(window=d_window).mean()

    # Drop intermediate columns
    df.drop(['ll', 'hh'], axis=1, inplace=True)

    return df

### K. Percentage of Price Change (PROC)

def proc(df, column_name='adj_close', window=1):
    """
    Calculate Percentage of Price Change (PROC) for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which PROC is calculated. Default is 'Close'.
    - window (int): Window size for PROC calculation. Default is 1.

    Returns:
    - pd.DataFrame: DataFrame with an added column for PROC.
    """
    # Calculate the percentage change in price using rolling window
    df['proc'] = df[column_name].pct_change().rolling(window=window).mean() * 100

    return df

### L. Momentum (MO(1))

def momentum(df, column_name='adj_close', window=1):
    """
    Calculate Momentum (MO) for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which Momentum is calculated. Default is 'Close'.
    - window (int): Window size for Momentum calculation. Default is 1.

    Returns:
    - pd.DataFrame: DataFrame with an added column for Momentum.
    """
    # Calculate the difference in price over the specified window
    df['momentum'] = df[column_name].diff(window)

    return df

### M. First-Order Lag (LAG(1))

def first_order_lag(df, column_name='adj_close', lag=1):
    """
    Calculate First-Order Lag (LAG(1)) for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column for which the lag is calculated. Default is 'Close'.
    - lag (int): Number of periods to lag. Default is 1.

    Returns:
    - pd.DataFrame: DataFrame with an added column for the First-Order Lag.
    """
    # Calculate the First-Order Lag using the shift() method
    df[f'lag_{lag}'] = df[column_name].shift(lag)

    return df

### N. Trading Volume (VOL)

def trading_volume(df, volume_column='volume'):
    """
    Calculate Trading Volume (VOL) for a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - volume_column (str): Name of the column containing trading volume. Default is 'Volume'.

    Returns:
    - pd.DataFrame: DataFrame with an added column for Trading Volume.
    """
    df['vol'] = df[volume_column]

    return df

###########
# SCALING #
###########

def scale_dataframe(df):
    """
    Scale a DataFrame using Standard scaling.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.

    Returns:
    - pd.DataFrame: Scaled DataFrame.
    """
    # Scale the selected columns
    scaler = StandardScaler()

    index_column = df.index

    # Check if 'label' column exists
    if 'label' in df.columns:
        label_column = df['label']
        df = df.drop(columns=['label'])
    else:
        label_column = None

    columns_to_scale = df.columns

    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=columns_to_scale)
    scaled_df.index = index_column

    # Re-add the 'label' column if it existed
    if label_column is not None:
        scaled_df['label'] = label_column

    return scaled_df


def train_test_split(df, test_size=0.2):
    """
    Split a time series dataset into training, testing, and validation sets.

    Parameters:
    - df: NumPy array or matrix, the input time series dataset.
    - test_size: Float, the proportion of the dataset to include in the test split.
    - val_size: Float, the proportion of the dataset to include in the validation split.

    Returns:
    - df_train, df_test: Pandas arrays, representing features and target values for each set.
    """

    # Extract index number of splitting points
    len_df = len(df)
    index_1 = round(len_df*(1-(test_size)))
    index_2 = index_1 +1

    # Extract values at previously calculated splitting points
    date_1 = df.index[index_1]
    date_2 = df.index[index_2]

    # Construct train_df, val_df and test_df
    df_train = df[:date_1]
    df_test = df[date_2:]

    return df_train, df_test


def input_matrix_split_X_y(df, window_size=5):
    """
    Reshape a DataFrame into a 3D NumPy arrays (num_observations, window_size, num_features)

    Parameters:
    - df: DataFrame with a list of time series data
    - sequence_length: the number of time steps to consider for each observation

    Returns:
    - X, y: a 3D NumPy arrays, one for the features and one for the lables
    """
    df_np = df.to_numpy()
    X = []
    y = []

    df_X = df.drop('label', axis=1)
    df_y = df['label']

    for i in range(len(df_np)-(window_size)):
        row = df_X[i:i+window_size]
        X.append(row)
        label = df_y[i+(window_size)]
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    y = np.expand_dims(y, axis=-1)

    return X, y
