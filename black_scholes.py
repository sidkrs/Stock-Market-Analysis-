import get_rf as rf
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
import datetime
 
RISK_FREE_RATE = rf.extract_treasury_rate(rf.URL)

def get_stock():
    ticker = input('Enter stock ticker: ')
    stock = yf.Ticker(ticker)
    expiration_dates = stock.options
    print(f'\nAvailable option expiration dates for {ticker}:')
    print(expiration_dates)
    
    option = int(input('\nWould you like to input 1. Maturity Date or 2. Days to Maturity? (Enter 1 or 2): '))
    if option == 1:
        mat_date = input('Enter maturity date (mm/dd/yyyy): ')
        current_date = datetime.date.today()
        maturity_date = datetime.datetime.strptime(mat_date, '%m/%d/%Y').date()
        difference = (maturity_date - current_date).days
        days = difference/365
    elif option == 2:
        difference = int(input('Enter days to maturity: '))
        days = difference/365
    
    maturity_date = datetime.date.today() + datetime.timedelta(days=difference)
    formatted_maturity_date = maturity_date.strftime('%Y-%m-%d')
    opts = stock.option_chain(formatted_maturity_date)
    call_strikes = opts.calls['strike']
    put_strikes = opts.puts['strike']
    strike_prices_df = pd.DataFrame({'Call ': call_strikes,' Put ': put_strikes})
    
    print('\nCurrent Price: $', stock.info['previousClose'])
    original_max_rows = pd.get_option('display.max_rows')
    pd.set_option('display.max_rows', None)
    print('Market Strike Prices:')
    print(strike_prices_df)
    pd.set_option('display.max_rows', original_max_rows)

    strike_price = float(input('\nEnter strike price: $'))
    return ticker, days, strike_price

def get_market_option_price(ticker, days, strike):
    stock = yf.Ticker(ticker)
    days = days * 365
    current_date = datetime.date.today()
    maturity_date = current_date + datetime.timedelta(days=days)
    formatted_maturity_date = maturity_date.strftime('%Y-%m-%d')
    option_chain = stock.option_chain(formatted_maturity_date)
    calls = option_chain.calls
    selected_calls = calls[calls['strike'] == strike]
    puts = option_chain.puts
    selected_puts = puts[puts['strike'] == strike]
    return selected_calls, selected_puts, formatted_maturity_date

def get_vals(ticker):
    tick = yf.Ticker(ticker)
    current_price = tick.info['previousClose']
    hist = tick.history(period='1y')
    std_dev = hist['Close'].std()
    sigma = std_dev / 100

    return current_price, sigma

def BS_CALL(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S/K) + (r + (sigma**2)/2)*T) / (sigma*(np.sqrt(T)))
    d2 = d1 - (sigma * np.sqrt(T))
    C = S * N(d1) - K * np.exp(-r*T)* N(d2)
    return C

def BS_PUT(S, K, T, r, sigma):
    N = norm.cdf
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma* np.sqrt(T)
    P = K*np.exp(-r*T)*N(-d2) - S*N(-d1)
    return P

def implied_volatility(m_call, m_put, S, K, T, r, initial_sigma):
    tolerance = 0.01
    max_iterations = 100
    sigma_call = initial_sigma
    sigma_put = initial_sigma

    for _ in range(max_iterations):
        e_call = BS_CALL(S, K, T, r, sigma_call)
        if abs(e_call - m_call) < tolerance:
            call_vol = round(sigma_call, 4)
            break
        elif e_call > m_call:
            sigma_call -= sigma_call * 0.05
        else:
            sigma_call += sigma_call * 0.05
    else:
        call_vol = None

    for _ in range(max_iterations):
        e_put = BS_PUT(S, K, T, r, sigma_put)
        if abs(e_put - m_put) < tolerance:
            put_vol = round(sigma_put, 4)
            break
        elif e_put > m_put:
            sigma_put -= sigma_put * 0.05
        else:
            sigma_put += sigma_put * 0.05
    else:
        put_vol = None

    return call_vol, put_vol


def plot_sigma_change(S, K, T, r):
    Sigmas = np.arange(0.01, 1.5, 0.01)
    calls = [BS_CALL(S, K, T, r, sig) for sig in Sigmas]
    puts = [BS_PUT(S, K, T, r, sig) for sig in Sigmas]
    plt.plot(Sigmas, calls, label='Call Value')
    plt.plot(Sigmas, puts, label='Put Value')
    plt.xlabel('Sigmas')
    plt.ylabel('Value')
    plt.title('Value of Option as Sigma Changes')
    plt.legend()  
    plt.show() # Corrected here

def plot_time_change(S, K, r, sigma):
    Ts = np.arange(0.0001, 1, 0.01)
    calls = [BS_CALL(S, K, T, r, sigma) for T in Ts]
    puts = [BS_PUT(S, K, T, r, sigma) for T in Ts]
    plt.plot(Ts, calls, label='Call Value')
    plt.plot(Ts, puts, label='Put Value')
    plt.xlabel('Time to Maturity')
    plt.ylabel('Value')
    plt.title('Value of Option as Time to Maturity Changes')
    plt.legend()
    plt.show() # Corrected here

def plot_implied_volatility(ticker, T, S, r, original_sigma, K):
    stock = yf.Ticker(ticker)
    formatted_maturity_date = (datetime.date.today() + datetime.timedelta(days=T * 365)).strftime('%Y-%m-%d')
    option_chain = stock.option_chain(formatted_maturity_date)
    calls = option_chain.calls
    puts = option_chain.puts
    current_price = S
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    added_call_label = False
    added_put_label = False

    for strike in calls['strike']:
        filtered_calls = calls[calls['strike'] == strike]
        if not filtered_calls.empty:
            m_call_price = filtered_calls['lastPrice'].iloc[0]
            call_vol, _ = implied_volatility(m_call_price, 0, S, strike, T, r, original_sigma)
            if not added_call_label:
                ax1.scatter(strike, call_vol, color='blue', label='Call Volatility')
                added_call_label = True
            else:
                ax1.scatter(strike, call_vol, color='blue')

    ax1.axhline(y=original_sigma, color='r', linestyle='-', label='Used Sigma')
    ax1.axvline(x=current_price, color='green', linestyle='--', label='Current Price')
    ax1.axvline(x=K, color='orange', linestyle='--', label='Chosen Strike Price')
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Sigma')
    ax1.set_title('Call Option Implied Volatility')
    ax1.legend()

    for strike in puts['strike']:
        filtered_puts = puts[puts['strike'] == strike]
        if not filtered_puts.empty:
            m_put_price = filtered_puts['lastPrice'].iloc[0]
            _, put_vol = implied_volatility(0, m_put_price, S, strike, T, r, original_sigma)
            if not added_put_label:
                ax2.scatter(strike, put_vol, color='green', label='Put Volatility')
                added_put_label = True
            else:
                ax2.scatter(strike, put_vol, color='green')

    ax2.axhline(y=original_sigma, color='r', linestyle='-', label='Used Sigma')
    ax2.axvline(x=current_price, color='green', linestyle='--', label='Current Price')
    ax2.axvline(x=K, color='orange', linestyle='--', label='Chosen Strike Price')
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Sigma')
    ax2.set_title('Put Option Implied Volatility')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_compare_prices_with_maturities(ticker, strike, S, r, sigma):
    stock = yf.Ticker(ticker)
    options_dates = stock.options

    market_call_prices = []
    bs_call_prices = []
    market_put_prices = []
    bs_put_prices = []
    valid_dates = []

    for date in options_dates:
        try:
            option_chain = stock.option_chain(date)
            calls = option_chain.calls
            puts = option_chain.puts

            filtered_call = calls[calls['strike'] == strike]
            filtered_put = puts[puts['strike'] == strike]

            if not filtered_call.empty and not filtered_put.empty:
                T = (datetime.datetime.strptime(date, '%Y-%m-%d').date() - datetime.date.today()).days / 365

                market_call_price = filtered_call['lastPrice'].iloc[0]
                market_call_prices.append(market_call_price)
                bs_call_prices.append(BS_CALL(S, strike, T, r, sigma))

                market_put_price = filtered_put['lastPrice'].iloc[0]
                market_put_prices.append(market_put_price)
                bs_put_prices.append(BS_PUT(S, strike, T, r, sigma))

                valid_dates.append(date)
        except:
            pass  # Handling cases where the strike price doesn't exist

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(valid_dates, market_call_prices, 'o-', label='Market Call Price')
    plt.plot(valid_dates, bs_call_prices, 'x-', label='BS Call Price')
    plt.xlabel('Maturity Date')
    plt.ylabel('Price')
    plt.title('Call Option Prices')
    plt.xticks(rotation=45)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_dates, market_put_prices, 'o-', label='Market Put Price')
    plt.plot(valid_dates, bs_put_prices, 'x-', label='BS Put Price')
    plt.xlabel('Maturity Date')
    plt.ylabel('Price')
    plt.title('Put Option Prices')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    ticker, T, K = get_stock()
    s, sigma = get_vals(ticker)
    r = RISK_FREE_RATE
    call_price = BS_CALL(s, K, T, r, sigma)
    put_price = BS_PUT(s, K, T, r, sigma)
    print('\nCalculated:')
    print(pd.DataFrame({'Call Price': [f"{call_price:.2f}"], ' Put Price': [f"{put_price:.2f}"]}))

    call, put, mat_date = get_market_option_price(ticker, T, K)
    print('\nMarket Option Prices for', ticker, 'with strike price', K, 'and expiration on', mat_date)
    
    if not call.empty:
        print('Call:')
        print(call[['strike', 'lastPrice']])
        m_call_price = call['lastPrice'].values[0]
        call_vol, _ = implied_volatility(m_call_price, 0, s, K, T, r, sigma)
    else:
        print('No call options available for this strike price.')
        call_vol = None

    if not put.empty:
        print('Put:')
        print(put[['strike', 'lastPrice']])
        m_put_price = put['lastPrice'].values[0]
        _, put_vol = implied_volatility(0, m_put_price, s, K, T, r, sigma)
    else:
        print('No put options available for this strike price.')
        put_vol = None

    print('\nImplied Volatility')
    print(pd.DataFrame({'Call': [call_vol], 'Put': [put_vol]}))

    plot_sigma_change(s, K, T, r)
    plot_time_change(s, K, r, sigma)
    plot_implied_volatility(ticker, T, s, r, sigma, K)
    plot_compare_prices_with_maturities(ticker, K, s, r, sigma)


if __name__ == '__main__':
    main()


