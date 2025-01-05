import pandas as pd
import datetime
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np
import pyotp
from SmartApi import SmartConnect
from logzero import logger
import urllib.request
import json

# yfinance override




target_allocation = 15000
# Credentials
api_key = 'fYXsEaUG'
username = 'V111503'
pwd = '1995'

# Token generation
try:
    token = "FPNYSK4M3YFZM6GF6VCIVYDFNI"
    totp = pyotp.TOTP(token).now()
except Exception as e:
    logger.error("Invalid Token: The provided token is not valid.")
    raise e

# API Connection
obj = SmartConnect(api_key=api_key)
try:
    data = obj.generateSession(username, pwd, totp)
    feed_token = obj.getfeedToken()
except Exception as e:
    logger.error(f"Error generating session: {e}")
    raise e

holdings = obj.holding()
print(holdings)

# Loop through and get trading symbols
trading_symbols = [item['tradingsymbol'] for item in holdings['data']]
print(trading_symbols)

# Remove the "-EQ" and "-BE" suffixes from each trading symbol
trading_symbols_cleaned = [symbol.replace('-EQ', '').replace('-BE', '').replace('-ST', '') for symbol in trading_symbols]
print(trading_symbols_cleaned)



# List of stocks to exclude from selling
excluded_stocks = [ '',
    'MID150BEES',  'ALFATRAN', 'GOLDBEES', 'CONART',
    'SJLOGISTIC', 'NIFTYBEES', 'CELLECOR', 'KPIGREEN', 'TTIL',
    'MIDSMALL', 'PRITI', 'JUNIORBEES', 'AIIL','STAR',  'SHAKTIPUMP', 'TEJASNET','SAILIFE', 'CELLECOR-SM', 'IGIL', 'JOSTS'
]





# Fetch instrument list
instrument_url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
try:
    response = urllib.request.urlopen(instrument_url)
    instrument_list = json.loads(response.read())

except Exception as e:
    logger.error(f"Failed to fetch instrument list: {e}")
    raise e

# Function to look up the token for a specific ticker
def token_lookup(ticker, instrument_list, exchange="NSE"):
    for instrument in instrument_list:
        if instrument["name"] == ticker and instrument["exch_seg"] == exchange and instrument["symbol"].split('-')[-1] == "EQ" and instrument["lotsize"]== "1":
            return instrument["token"]
    return None  # Return None if no matching token is found

# Function to get all symbols in the equity segment, removing the "-EQ" suffix
def get_all_equity_symbols(instrument_list, exchange="NSE"):
    symbols = [
        instrument["symbol"].replace("-EQ", "") for instrument in instrument_list
        if instrument.get("exch_seg") == exchange and instrument.get("symbol", "").endswith("-EQ")
    ]
    return symbols

# Example usage
equity_symbols = get_all_equity_symbols(instrument_list)
print(equity_symbols)  # Prints array of all equity symbols without the "-EQ" suffix

tickers_a = pd.DataFrame({'Symbol': equity_symbols})
index_name = '^CRSLDX'  # NIFTY500
start_date = datetime.datetime.now() - datetime.timedelta(days=400)
end_date = datetime.date.today()

tickers = pd.read_excel('C:\\Users\\Venkat Jaswanth\\Downloads\\MCAP28032024.xlsx')
last_column_name = tickers.columns[-1]  # Get the name of the last column

tickers[last_column_name] = pd.to_numeric(tickers[last_column_name], errors='coerce')
shakti_pumps_row = tickers[tickers.apply(lambda row: row.astype(str).str.contains('BCC FUBA', case=False, na=False).any(), axis=1)]

# Display the result
print(shakti_pumps_row)
tickers = tickers[tickers[last_column_name] > 100000]
tickers = tickers[tickers['Symbol'].isin(equity_symbols)]

# Initialize export list
exportList = pd.DataFrame(columns=[
    'Stock', '20 Day MA', '50 Day MA', '150 Day MA', 
    '200 Day MA', '52 Week Low', '52 Week High', 'CP', 'AVG SHARPE','RS_Rating'
])
returns_multiples = []

# Index Returns
index_df = yf.download(index_name, start_date, end_date)
index_df['Percent Change'] = index_df['Adj Close'].pct_change()
index_return = (index_df['Percent Change'] + 1).cumprod().iloc[-1]

def momentum_func(the_array):
    r = np.log(the_array)
    slope, _, rvalue, _, _ = linregress(np.arange(len(r)), r)
    annualized = (1 + slope) ** 252
    return annualized * (rvalue ** 2)

def check_upper_circuit(data, consecutive_days=3):
    # Identify potential upper circuit days (when the Close equals the High)
    upper_circuit_days = (data['Close'] == data['High']).astype(int)
    
    # Check for consecutive occurrences of upper circuit days
    consecutive_upper_circuit = (upper_circuit_days.rolling(window=consecutive_days).sum() == consecutive_days).any()
    
    return consecutive_upper_circuit

# Function to calculate Sharpe ratio
def calculate_sharpe(daily_returns):
    risk_free_rate = 0.06 / 252  # Assuming a 6% annual risk-free rate
    excess_returns = daily_returns - risk_free_rate
    sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns, axis=0)
    return sharpe_ratio * np.sqrt(252)


# Process each ticker
for ticker in tickers['Symbol']:
    try:
        ticker = ticker + ".NS"
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # Calculate returns relative to the index
        df['Percent Change'] = df['Adj Close'].pct_change()
        stock_return = (df['Percent Change'] + 1).cumprod().iloc[-1]
        returns_multiple = round((stock_return / index_return), 2)
        returns_multiples.append(returns_multiple)
        
        print(f'Ticker: {ticker}; Returns Multiple against NIFTY50: {returns_multiple}')
    except Exception as e:
        print(f"Could not gather data on {ticker}. Error: {e}")

# Create DataFrame for top 30% performing stocks
rs_df = pd.DataFrame(zip(tickers['Symbol'], returns_multiples), columns=['Ticker', 'Returns_multiple'])
rs_df['RS_Rating'] = rs_df['Returns_multiple'].rank(pct=True) * 100
rs_df = rs_df[rs_df['RS_Rating'] >= rs_df['RS_Rating'].quantile(0.70)]

# Checking Minervini conditions
for stock in  rs_df['Ticker']:
    try:
        df = yf.download(stock + '.NS', start_date, end_date)
        sma = [20, 50, 150, 200]
        for x in sma:
            df[f"SMA_{x}"] = round(df['Adj Close'].rolling(window=x).mean(), 2)
        
      




        # Storing required values
        currentClose = df["Adj Close"].iloc[-1].values[0]
        moving_average_20 = df["SMA_20"].iloc[-1]
        moving_average_50 = df["SMA_50"].iloc[-1]
        moving_average_150 = df["SMA_150"].iloc[-1]
        moving_average_200 = df["SMA_200"].iloc[-1]
        low_of_52week = round(df["Low"].iloc[-252:].min(), 2).values[0]
        high_of_52week = round(df["High"].iloc[-252:].min(), 2).values[0]
        RS_Rating = round(rs_df.loc[rs_df['Ticker'] == stock, 'RS_Rating'].values[0])
        # Calculate daily returns
        df['Daily Return'] = df['Adj Close'].pct_change()
        
        
        # Calculate Sharpe Ratio
        sharpe_ratio = calculate_sharpe(df['Daily Return'].dropna())


        # Calculate % away from 52-week high
        pct_away_52w_high = ((high_of_52week - currentClose) / high_of_52week) * 100

        try:
            moving_average_200_20 = df["SMA_200"].iloc[-20]
        except Exception:
            moving_average_200_20 = 0

        # Minervini conditions
        conditions = [
            currentClose > moving_average_150 > moving_average_200,
            moving_average_150 > moving_average_200,
            moving_average_200 > moving_average_200_20,
            moving_average_50 > moving_average_150 > moving_average_200,
            currentClose > moving_average_50,
            currentClose >= (1.3 * low_of_52week),
            currentClose >= (.75 * high_of_52week),
            currentClose < 10000
           
        ]

        # Add to export list if all conditions are met
        if all(conditions):
            new_row = pd.DataFrame([{
        'Stock': stock,  "20 Day MA": moving_average_20,
        "50 Day MA": moving_average_50, "150 Day MA": moving_average_150,
        "200 Day MA": moving_average_200, "52 Week Low": low_of_52week,
        "52 Week High": high_of_52week, "CP": currentClose,"RS_Rating": RS_Rating
        }])
    
            exportList = pd.concat([exportList, new_row], ignore_index=True)
    except Exception as e:
        print(f"Could not gather data on {stock}: {e}")

# Sort and prepare the final list

MomentumStocks = exportList
MomentumStocks['RS_Rating'] = MomentumStocks['RS_Rating'].rank(ascending=False)

# Sort the stocks by the combined rank
MomentumStocks = MomentumStocks.sort_values(by='RS_Rating', ascending=True)
MomentumStocks = MomentumStocks[['Stock', 'CP']]
MomentumStocks['Quantity'] = (target_allocation / MomentumStocks['CP']).round(2).astype(int)  # Calculate quantity
MomentumStocks['Purchase Value'] = (MomentumStocks['Quantity'] * MomentumStocks['CP']).round(2)
MomentumStocks= MomentumStocks.head(50)
# Display top 25 stocks
print(MomentumStocks)






# Determine stocks to sell (those not in the top 25 momentum stocks or in the excluded list)
stocks_to_sell = [symbol for symbol in trading_symbols_cleaned 
                  if symbol not in MomentumStocks['Stock'].values and symbol not in excluded_stocks]



new_stocks_to_buy = MomentumStocks[~MomentumStocks['Stock'].isin(trading_symbols_cleaned)].head(len(stocks_to_sell))

if not new_stocks_to_buy.empty:
    print("The following momentum stocks are not in your PF:")
    print(new_stocks_to_buy)
    print(stocks_to_sell)
else:
    print("All top 25 momentum stocks are present in the trading symbols list.")



# Sell existing stocks not in top 25
for symbol in stocks_to_sell:
    symbol_token = token_lookup(symbol, instrument_list)
    
    if symbol_token is None:
        logger.warning(f"Token not found for {symbol}. Skipping sell...")
        continue
    
    # Assuming a market sell order for the full quantity held
    orderparams = {
        "variety": "NORMAL",
        "tradingsymbol": f"{symbol}-EQ",
        "symboltoken": symbol_token,
        "transactiontype": "SELL",
        "exchange": "NSE",
        "ordertype": "MARKET",
        "producttype": "DELIVERY",
        "duration": "DAY",
        "quantity": str(int(holdings['data'][trading_symbols.index(symbol+"-EQ")]['quantity'])),  # Quantity is fetched from holdings
    }
    
    try:
        response = obj.placeOrder(orderparams)
        logger.info(f"Sold {symbol}: {response}")
    except Exception as e:
        logger.exception(f"Order placement failed for selling {symbol}: {e}")



if not new_stocks_to_buy.empty:
    # Place orders for all new stock symbols
    for _, row in new_stocks_to_buy.iterrows():
        symbol = row['Stock']
        try:
            symbol_token = token_lookup(symbol, instrument_list)
            
            if symbol_token is None:
                logger.warning(f"Token not found for {symbol}. Skipping buy...")
                continue
            
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": f"{symbol}-EQ",
                "symboltoken": symbol_token,
                "transactiontype": "BUY",
                "exchange": "NSE",
                "ordertype": "MARKET",
                "producttype": "DELIVERY",
                "duration": "DAY",
                "quantity": str(int(row['Quantity'])),  # Quantity is an integer
            }
            
            # Place the order and log the response
            response = obj.placeOrder(orderparams)
            logger.info(f"Placed order for {symbol}: {response}")
            
        except Exception as e:
            logger.exception(f"Order placement failed for buying {symbol}: {e}")
else:
    print("All top 25 momentum stocks are already in your holdings. No new orders will be placed.")
