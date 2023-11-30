import ticker_sentiment as ts
import black_scholes as bs
import gpt_ai as gpt
import stock_pred as sp
import get_rf as rf
import pandas as pd
from art import text2art

def ticker_sentiment():
    basket = ts.get_basket()
    basket = [ticker.upper() for ticker in basket]
    pcas = []
    for i in basket:
        pcas.append(ts.get_tick_sent(i)[0])
    
    apds = ts.apd_stock(basket, pcas)
    most_volatile = max(apds, key=apds.get)
    print(apds)
    print(f'The most volatile stock is {most_volatile} with an APD of {apds.get(most_volatile)}')
    print(f'The least volatile stock is {min(apds, key=apds.get)} with an APD of {apds.get(min(apds, key=apds.get))}')
    ts.plot(apds)

def black_scholes():
    ticker, T, K = bs.get_stock()
    s, sigma = bs.get_vals(ticker)
    r = rf.extract_treasury_rate(rf.URL)
    call_price = bs.BS_CALL(s, K, T, r, sigma)
    put_price = bs.BS_PUT(s, K, T, r, sigma)
    print('\nCalculated:')
    print(pd.DataFrame({'Call Price': [f'{call_price:.2f}'], ' Put Price': [f'{put_price:.2f}']}))

    call, put, mat_date = bs.get_market_option_price(ticker, T, K)
    print('\nMarket Option Prices for', ticker, 'with strike price', K, 'and expiration on', mat_date)
    
    if not call.empty:
        print('Call:')
        print(call[['strike', 'lastPrice']])
        m_call_price = call['lastPrice'].values[0]
        call_vol, _ = bs.implied_volatility(m_call_price, 0, s, K, T, r, sigma)
    else:
        print('No call options available for this strike price.')
        call_vol = None

    if not put.empty:
        print('Put:')
        print(put[['strike', 'lastPrice']])
        m_put_price = put['lastPrice'].values[0]
        _, put_vol = bs.implied_volatility(0, m_put_price, s, K, T, r, sigma)
    else:
        print('No put options available for this strike price.')
        put_vol = None

    print('\nImplied Volatility')
    print(pd.DataFrame({'Call': [call_vol], 'Put': [put_vol]}))

    bs.plot_sigma_change(s, K, T, r)
    bs.plot_time_change(s, K, r, sigma)
    bs.plot_implied_volatility(ticker, T, s, r, sigma, K)
    bs.plot_compare_prices_with_maturities(ticker, K, s, r, sigma)

def chat_gpt():
    print('Ask a question regarding a sector in the stock market')
    print('')
    user_input = input('Enter a message: ')
    completion = gpt.chat(user_input)
    print(completion.choices[0].message.content.replace('\n', ' '))

def stock_pred():
    df = sp.get_stock()
    sp.pred(df)

def main():
    line1 = text2art("Welcome to the") 
    line2 = text2art("Stock Market")    
    line3 = text2art("Analysis Tool!") 
    combined_art = line1 + '\n' + line2 + '\n' + line3
    print(combined_art)
    print('\nWhat would you like to do?')
    print('\n1. Get the sentiment of the market')
    print('2. Get the sentiment of a basket of stocks')
    print('3. Predict the price of a stock')
    print('4. Calculate the price of an option')
    print('5. Advice from an AI financial advisor')
    print('6. Exit')
    choice = input('\nEnter a number: ')
    while choice != '6':
        if choice == '1':
            #market_sentiment()
            print('not done yet :)')
        elif choice == '2':
            ticker_sentiment()
        elif choice == '3':
            stock_pred()
        elif choice == '4':
            black_scholes()
        elif choice == '5':
            chat_gpt()
        else:
            print('Invalid input.')
        print('\n\nWhat would you like to do?')
        print('\n1. Get the sentiment of the market')
        print('2. Get the sentiment of a basket of stocks')
        print('3. Predict the price of a stock')
        print('4. Calculate the price of an option')
        print('5. Advice from an AI financial advisor')
        print('6. Exit')
        choice = input('\nEnter a number: ')
    print('Goodbye!')

if __name__ == '__main__':
    main()
