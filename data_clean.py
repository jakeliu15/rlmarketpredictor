import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl import config_tickers
from finrl.config import INDICATORS
import itertools

# Define date ranges
TRAIN_START_DATE = '2009-01-01'
TRAIN_END_DATE = '2020-07-01'
TRADE_START_DATE = '2020-07-01'
TRADE_END_DATE = '2023-05-01'

# Define symbols
symbols = [
    'aapl', 'msft', 'meta', 'ibm', 'hd', 'cat', 'amzn', 'intc', 't', 'v', 'gs'
]

# Download data
df_raw = YahooDownloader(start_date=TRAIN_START_DATE,
                         end_date=TRADE_END_DATE,
                         ticker_list=symbols).fetch_data()
print(df_raw.head())

# Feature engineering
fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list=INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature=False)

processed = fe.preprocess_data(df_raw)
print(processed)

# Create full processed dataframe
list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(), processed['date'].max()).astype(str))
combination = list(itertools.product(list_date, list_ticker))
processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"], how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date', 'tic'])
processed_full = processed_full.fillna(0)
print(processed_full)

# Split data into train and trade sets
train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
print(f"Length of train data: {len(train)}")
print(f"Length of trade data: {len(trade)}")

# Save to CSV files
train_path = 'train_data.csv'
trade_path = 'trade_data.csv'

train.to_csv(train_path, index=False, encoding='utf-8-sig')
trade.to_csv(trade_path, index=False, encoding='utf-8-sig')

print(f"Train data saved to {train_path}")
print(f"Trade data saved to {trade_path}")

