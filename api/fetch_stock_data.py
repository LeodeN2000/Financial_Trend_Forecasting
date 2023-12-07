import yfinance as yf
import pandas as pd

def get_stock_data(ticker="btc-usd"):

    stock = yf.Ticker(ticker)

    # get historical market data
    hist = stock.history(period="1mo", interval='60m')

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

# if __name__ == "__main__":

#     print(get_stock_data("btc-usd"))
