import os
import sys
import pickle
import glob

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

def seconds_until_midnight(last):
    
    tomorrow = last + datetime.timedelta(1)
    midnight = datetime.datetime(year = tomorrow.year, month = tomorrow.month, day = tomorrow.day,
                                hour = 0, minute = 0, second = 0)
    return (midnight - last).seconds

def seconds_since_midnight():
    
    now = datetime.datetime.now()
    ssm = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
    return int(ssm)

def generate_amount():
    
    mean_amount = np.random.uniform(5, 100)
    std_amount = mean_amount/2
    
    amount = np.random.normal(mean_amount, std_amount)
                    
    # If amount negative, draw from a uniform distribution
    if amount < 0:
        amount = np.random.uniform(0, mean_amount*2)
        
    amount = np.round(amount, decimals=2)
    
    return amount

def generate_single_transaction(new_id, new_sec, new_day, todays_date, N_CUSTOMERS, N_TERMINAL, latest_data):
    
    new_customer = random.choice(latest_data['CUSTOMER_ID'].values)
    new_terminal = random.choice(latest_data['TERMINAL_ID'].values)
    new_amount = generate_amount()
    
    new_transaction = {'TRANSACTION_ID': new_id,
                       'TX_DATETIME': pd.to_datetime(new_sec, unit='s', origin=todays_date),
                       'CUSTOMER_ID': new_customer,
                       'TERMINAL_ID': new_terminal,
                       'TX_AMOUNT': new_amount,
                       'TX_TIME_SECONDS': new_sec,
                       'TX_TIME_DAYS': new_day}
    
    return new_transaction

