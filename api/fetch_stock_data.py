import yfinance as yf
import pandas as pd

def get_stock_data(ticker="btc-usd", time_horizon='60m'):

    stock = yf.Ticker(ticker)
    if time_horizon == '5m':
        period="1d"
    else:
        period="1mo"
    # get historical market data
    hist = stock.history(period=period, interval=time_horizon)

    volumes = hist['Volume']

    open_prices = hist['Open']

    close_prices = hist['Close']

    high_prices = hist['High']

    low_prices = hist['Low']

    hist_df = pd.DataFrame(data={
                    'volume': volumes,
                    'open': open_prices,
                    'high': high_prices,
                    'low':low_prices,
                    'adj_close':close_prices})


    return hist_df

if __name__ == "__main__":

     print(get_stock_data("btc-usd"))
