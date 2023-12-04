import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly

def get_stock():
    '''
    Parameters: None
    Returns: a dataframe of stock data
    Does: Prompts the user to enter a ticker and returns a dataframe of the stock's data.
    '''
    ticker = input("Enter a ticker: ").upper()
    ticker_data = yf.Ticker(ticker)
    # Fetch data from the maximum available history
    ticker_df = ticker_data.history(start='2020-1-1')
    ticker_df.reset_index(inplace=True)
    
    # Convert 'Date' column to datetime format
    ticker_df['Date'] = pd.to_datetime(ticker_df['Date']).dt.date
    
    # Prepare DataFrame for Prophet
    df = ticker_df[['Date', 'Close']]
    df.columns = ['ds', 'y']
    return df, ticker

def pred(df, ticker):
    '''
    Parameters: a dataframe of stock data
    Returns: None
    Does: Takes a dataframe of stock data and plots a forecast of the stock's price.
    '''
    # Fit the model
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(df)
    future = prophet.make_future_dataframe(periods=365)
    predictions = prophet.predict(future)

    # Plot the forecast
    fig = plot_plotly(prophet, predictions)
    # Update axis titles and add a plot title
    fig.update_layout(
    xaxis_title="Year",
    yaxis_title="Price (USD)",
    title=f"Stock Price Forecast for {ticker}"
    )
    fig.show()
    fig_components = prophet.plot_components(predictions)
    fig_components.show()



def main():
    df, ticker = get_stock()
    pred(df, ticker)

if __name__ == "__main__":
    main()
