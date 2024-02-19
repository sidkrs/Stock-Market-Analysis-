import ticker_sentiment as ts
import black_scholes as bs
import gpt_ai as gpt
import stock_pred as sp
import get_rf as rf
import pandas as pd
from art import text2art

def market_sentiment():
    files = fs.get_filenames(fs.SPEECH_DIR)
    ordered_files = sorted(files, key=fs.get_date_fromfile)
    dates = fs.store_dates(ordered_files)
    sentiment_scores = fs.sentiment_scores_pdfs(ordered_files)
    normalized_sentscores = fs.normalize(sentiment_scores)
    df = fs.extract_prices(fs.INDEX_DIR)
    filtered_df = df[df['Date'].isin(dates)]
    roc_df = fs.df_roc(filtered_df)
    roc_df['Polarity Score'] = normalized_sentscores
    corr_df = roc_df.iloc[:-1]
    
    # Calculate the correlation between the sentiment scores and the rate of change
    corr_spy = corr_df['Polarity Score'].corr(corr_df['SPY Rate of Change'])
    corr_dow = corr_df['Polarity Score'].corr(corr_df['DIA Rate of Change'])
    corr_vix = corr_df['Polarity Score'].corr(corr_df['VIX Rate of Change'])
    corr_qqq = corr_df['Polarity Score'].corr(corr_df['QQQ Rate of Change'])
    
    # Print the correlation values
    print(f"Correlation with S&P500: {corr_spy}\n")
    print(f"Correlation with DOW: {corr_dow}\n")
    print(f"Correlation with NASDAQ: {corr_qqq}\n")
    print(f"Correlation with VIX: {corr_vix}\n")
    
    # Plot the scatter plots
    plot = fs.plot_scatter
    plot(corr_df['Polarity Score'], corr_df['SPY Rate of Change'], 'Fed Speech Sentiment Score vs. S&P500 Rate of Change', 'Normalized Fed Speech Sentiment Scores', '% Change in Index')
    plot(corr_df['Polarity Score'], corr_df['DIA Rate of Change'], 'Fed Speech Sentiment Score vs. DOW Rate of Change', 'Normalized Fed Speech Sentiment Scores', '% Change in Index')
    plot(corr_df['Polarity Score'], corr_df['QQQ Rate of Change'], 'Fed Speech Sentiment Score vs. NASDAQ Rate of Change', 'Normalized Fed Speech Sentiment Scores', '% Change in Index')
    plot(corr_df['Polarity Score'], corr_df['VIX Rate of Change'], 'Fed Speech Sentiment Score vs. VIX Rate of Change', 'Normalized Fed Speech Sentiment Scores', '% Change in Index')


def ticker_sentiment():
    '''
    Parameters: None
    Returns: None
    Does: Gets the sentiment of a basket of stocks and plots the results.
    '''
    # Get the basket of stocks
    basket = ts.get_basket()
    basket = [ticker.upper() for ticker in basket]
    pcas = []
    for i in basket:
        pcas.append(ts.get_tick_sent(i)[0])
    
    # Get the average pairwise distances
    apds = ts.apd_stock(basket, pcas)
    most_volatile = max(apds, key=apds.get)
    print(pd.DataFrame(list(apds.items()), columns=['TICKER', 'APD']))
    ts.plot_all_apds(apds)

    print(f'The most volatile stock is {most_volatile} with an APD of {apds.get(most_volatile)}')
    print(f'The least volatile stock is {min(apds, key=apds.get)} with an APD of {apds.get(min(apds, key=apds.get))}')
    ts.plot(apds)

def black_scholes():
    '''
    Parameters: None
    Returns: None
    Does: Calculates the price of an option using the Black-Scholes model and compares it to the market price.
    '''

    # Get the stock data
    ticker, T, K = bs.get_stock()
    s, sigma = bs.get_vals(ticker)
    r = rf.extract_treasury_rate(rf.URL)
    call_price = bs.BS_CALL(s, K, T, r, sigma)
    put_price = bs.BS_PUT(s, K, T, r, sigma)
    print('\nCalculated:')
    print(pd.DataFrame({'Call Price': [f'{call_price:.2f}'], ' Put Price': [f'{put_price:.2f}']}))

    # Get the market data
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

    # Print the implied volatility
    print('\nImplied Volatility')
    print(pd.DataFrame({'Call': [call_vol], 'Put': [put_vol]}))

    # Plot the results
    bs.plot_sigma_change(s, K, T, r)
    bs.plot_time_change(s, K, r, sigma)
    bs.plot_implied_volatility(ticker, T, s, r, sigma, K)
    bs.plot_compare_prices_with_maturities(ticker, K, s, r, sigma)

def chat_gpt():
    '''
    Parameters: None
    Returns: None
    Does: Uses the GPT-4 chat model to provide advice to the user.
    '''
    
    # Get the user's input
    print('Ask a question regarding a sector in the stock market or information about a specific stock.')
    print('')
    user_input = input('Enter a message: ')
    completion = gpt.chat(user_input)
    # Print the response, removing newline characters
    print(completion.choices[0].message.content.replace('\n', ' '))

def stock_pred():
    '''
    Parameters: None
    Returns: None
    Does: Predicts the price of a stock using Facebook Prophet.
    '''
    # Get the stock data
    df, ticker = sp.get_stock()
    # Predict the price
    sp.pred(df, ticker)

def main():
    # Print the welcome message
    line1 = text2art("Welcome to the") 
    line2 = text2art("Stock Market")    
    line3 = text2art("Analysis Tool!") 
    combined_art = line1 + '\n' + line2 + '\n' + line3
    
    # Print the menu
    print(combined_art)
    print('\nWhat would you like to do?')
    print('\n1. Get the sentiment of the market')
    print('2. Get the sentiment of a basket of stocks')
    print('3. Predict the price of a stock')
    print('4. Calculate the price of an option')
    print('5. Advice from an AI financial advisor')
    print('6. Exit')
    
    # Get the user's choice
    choice = input('\nEnter a number: ')
    while choice != '6':
        # Depending on choice run the function
        if choice == '1':
            print("\n")
            market_sentiment()
        elif choice == '2':
            print("\n")
            ticker_sentiment()
        elif choice == '3':
            print("\n")
            stock_pred()
        elif choice == '4':
            print("\n")
            black_scholes()
        elif choice == '5':
            print("\n")
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
