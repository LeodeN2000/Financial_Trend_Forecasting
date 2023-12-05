import ccxt
import pandas as pd

def get_btc_data():
    cex_x = ccxt.binance().fetch_ohlcv('BTC/USDT', '1h', limit = 50)

    # cex_x is a list of 500 items, one for every hour, on the hour.
    #
    # Each item has a list of 6 entries:
    # (0) timestamp (1) open price (2) high price (3) low price (4) close price (5) volume
    # Example item: [1662998400000, 1706.38, 1717.87, 1693, 1713.56, 2186632.9224]
    # Timestamp is unix time, but in ms. To get unix time (in s), divide by 1000

    # Example: get unix timestamps
    uts = [xi[0]/1000 for xi in cex_x]

    # Example: get open prices
    open_prices = [xi[1] for xi in cex_x]

    # Example: get high prices
    high_prices = [xi[2] for xi in cex_x]

    # Example: get low prices
    low_prices = [xi[3] for xi in cex_x]

    # Example: get close prices
    close_prices = [xi[4] for xi in cex_x]

    volumes = [xi[5] for xi in cex_x]

    df_btc = pd.DataFrame(data={'uts':uts,
                    'volume': volumes,
                    'open': open_prices,
                    'high': high_prices,
                    'low':low_prices,
                    'adj_close':close_prices})


    filtered_df_btc = df_btc[['open', 'high', 'low', 'adj_close', 'volume']]

    return filtered_df_btc
