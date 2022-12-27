import os
import sys
import pickle
import glob
import re
from dateutil import parser

import numpy as np
import pandas as pd

import datetime
import time

import random

def get_latest_data(DIR_RAW):

    list_of_files = glob.glob(DIR_RAW+'*')
    latest_file = max(list_of_files, key = os.path.getctime)

    with open(r"{}".format(latest_file), "rb") as input_file:
        latest_data = pickle.load(input_file)
        
    return latest_data

def get_dictionaries(latest_proc, customer_columns, terminal_columns):
    
    cdf = latest_proc.drop_duplicates('CUSTOMER_ID', keep = 'last')
    tdf = latest_proc.drop_duplicates('TERMINAL_ID', keep = 'last')
    
    cdf = cdf[customer_columns]
    tdf = tdf[terminal_columns]
    
    customer_dict = cdf.set_index('CUSTOMER_ID').T.to_dict('list')
    terminal_dict = tdf.set_index('TERMINAL_ID').T.to_dict('list')
    
    return customer_dict, terminal_dict

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

def convert_to_datetime(s):

    pattern = re.compile('\((.*?)\)')
    pattern.findall(s)[0]
    return parser.parse(pattern.findall(s)[0])

def preprocess_transaction(transactions, DIR_PROC, customer_columns, terminal_columns):
    
    latest_proc = get_latest_data(DIR_PROC)
    customer_dict, terminal_dict = get_dictionaries(latest_proc, customer_columns, terminal_columns)

    transactions = {k:[v] for k,v in transactions.items()}
    transactions = pd.DataFrame(transactions)
    
    transactions['TX_FRAUD'] = 0
    transactions['TX_FRAUD_SCENARIO'] = 0
    
    transactions['TX_DATETIME'] = transactions.TX_DATETIME.apply(lambda x: convert_to_datetime(x))

    transactions['TX_DURING_WEEKEND']=transactions.TX_DATETIME.apply(is_weekend)
    transactions['TX_DURING_NIGHT']=transactions.TX_DATETIME.apply(is_night)
    
    transactions = transactions.merge(pd.DataFrame(customer_dict, index = customer_columns[1:]).T, left_on = 'CUSTOMER_ID', right_index = True)
    transactions = transactions.merge(pd.DataFrame(terminal_dict, index = terminal_columns[1:]).T, left_on = 'TERMINAL_ID', right_index = True)
    
    return transactions