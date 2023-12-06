import yfinance as yf
import pandas as pd

def get_stock_data(ticker="appl"):

    stock = yf.Ticker(ticker)

    # get historical market data
    hist = stock.history(period="1mo")

    volumes = hist['Volumes']

    open_prices = hist['Open']

    close_prices = hist['Close']

    high_prices = hist['High']

    low_prices = hist['Low']

    hist_df = pd.DataFrame(data={'date': hist['Date'],
                    'volume': volumes,
                    'open': open_prices,
                    'high': high_prices,
                    'low':low_prices,
                    'adj_close':close_prices})


    return hist_df
