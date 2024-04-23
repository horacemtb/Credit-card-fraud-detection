import os
import sys
import pickle

import numpy as np
import pandas as pd

import datetime
import time

import random

def is_weekend(tx_datetime):
    
    # Transform date into weekday (0 is Monday, 6 is Sunday)
    weekday = tx_datetime.weekday()
    # Binary value: 0 if weekday, 1 if weekend
    is_weekend = weekday>=5
    
    return int(is_weekend)

def is_night(tx_datetime):
    
    # Get the hour of the transaction
    tx_hour = tx_datetime.hour
    # Binary value: 1 if hour less than 6, and 0 otherwise
    is_night = tx_hour<=6
    
    return int(is_night)

def get_customer_spending_behaviour_features(customer_transactions, windows_size_in_days=[1,7,30]):
    
    # Let us first order transactions chronologically
    customer_transactions=customer_transactions.sort_values('TX_DATETIME')
    
    # The transaction date and time is set as the index, which will allow the use of the rolling function 
    customer_transactions.index=customer_transactions.TX_DATETIME
    
    # For each window size
    for window_size in windows_size_in_days:
        
        # Compute the sum of the transaction amounts and the number of transactions for the given window size
        SUM_AMOUNT_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').sum()
        NB_TX_WINDOW=customer_transactions['TX_AMOUNT'].rolling(str(window_size)+'d').count()
    
        # Compute the average transaction amount for the given window size
        # NB_TX_WINDOW is always >0 since current transaction is always included
        AVG_AMOUNT_TX_WINDOW=SUM_AMOUNT_TX_WINDOW/NB_TX_WINDOW
    
        # Save feature values
        customer_transactions['CUSTOMER_ID_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_AVG_AMOUNT_'+str(window_size)+'DAY_WINDOW']=list(AVG_AMOUNT_TX_WINDOW)
    
    # Reindex according to transaction IDs
    customer_transactions.index=customer_transactions.TRANSACTION_ID
        
    # And return the dataframe with the new features
    return customer_transactions

def get_count_risk_rolling_window(terminal_transactions, delay_period=7, windows_size_in_days=[1,7,30], feature="TERMINAL_ID"):
    
    terminal_transactions=terminal_transactions.sort_values('TX_DATETIME')
    
    terminal_transactions.index=terminal_transactions.TX_DATETIME
    
    NB_FRAUD_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').sum()
    NB_TX_DELAY=terminal_transactions['TX_FRAUD'].rolling(str(delay_period)+'d').count()
    
    for window_size in windows_size_in_days:
    
        NB_FRAUD_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').sum()
        NB_TX_DELAY_WINDOW=terminal_transactions['TX_FRAUD'].rolling(str(delay_period+window_size)+'d').count()
    
        NB_FRAUD_WINDOW=NB_FRAUD_DELAY_WINDOW-NB_FRAUD_DELAY
        NB_TX_WINDOW=NB_TX_DELAY_WINDOW-NB_TX_DELAY
    
        RISK_WINDOW=NB_FRAUD_WINDOW/NB_TX_WINDOW
        
        terminal_transactions[feature+'_NB_TX_'+str(window_size)+'DAY_WINDOW']=list(NB_TX_WINDOW)
        terminal_transactions[feature+'_RISK_'+str(window_size)+'DAY_WINDOW']=list(RISK_WINDOW)
        
    terminal_transactions.index=terminal_transactions.TRANSACTION_ID
    
    # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0) 
    terminal_transactions.fillna(0,inplace=True)
    
    return terminal_transactions

def read_from_files(DIR_INPUT):
    
    files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT)]

    frames = []
    
    for f in files:
        df = pd.read_pickle(f)
        frames.append(df)
        del df
        
    df_final = pd.concat(frames)
    
    df_final=df_final.sort_values('TRANSACTION_ID')
    df_final.reset_index(drop=True,inplace=True)
    df_final=df_final.replace([-1],0)
    
    return df_final

def preprocess_transactions(transactions_df):
    
    transactions_df['TX_DURING_WEEKEND']=transactions_df.TX_DATETIME.apply(is_weekend)
    transactions_df['TX_DURING_NIGHT']=transactions_df.TX_DATETIME.apply(is_night)
    
    transactions_df=transactions_df.groupby('CUSTOMER_ID').apply(lambda x: get_customer_spending_behaviour_features(x, windows_size_in_days=[1,7,30]))
    transactions_df=transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    
    transactions_df=transactions_df.groupby('TERMINAL_ID').apply(lambda x: get_count_risk_rolling_window(x, delay_period=7, windows_size_in_days=[1,7,30], feature="TERMINAL_ID"))
    transactions_df=transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    
    return transactions_df

def save_processed(transactions_df, DIR_OUTPUT):
    
#     if not os.path.exists(DIR_OUTPUT):
#         os.makedirs(DIR_OUTPUT)

    start_date = transactions_df.loc[0, 'TX_DATETIME']
    start_date = datetime.datetime.strptime(f"{start_date.year}-{start_date.month}-{start_date.day}", "%Y-%m-%d")

    for day in range(transactions_df.TX_TIME_DAYS.max() - transactions_df.TX_TIME_DAYS.min()+1):
        
        transactions_day = transactions_df[transactions_df.TX_TIME_DAYS==transactions_df.TX_TIME_DAYS.min()+day].sort_values('TX_TIME_SECONDS')

        date = start_date + datetime.timedelta(days=day)
        filename_output = date.strftime("%Y-%m-%d")+'.pkl'

        # Protocol=4 required for Google Colab
        transactions_day.to_pickle(DIR_OUTPUT+filename_output, protocol=4)

def get_preprocessed_data(days, INPUT_DIR, OUTPUT_DIR):
    
    transactions_df = read_from_files(INPUT_DIR)
    start_date = transactions_df.iloc[-1, :]['TX_DATETIME'] - datetime.timedelta(days=days)
    transactions_df = transactions_df[transactions_df.TX_DATETIME > start_date]
    transactions_df = preprocess_transactions(transactions_df)
    save_processed(transactions_df, OUTPUT_DIR)