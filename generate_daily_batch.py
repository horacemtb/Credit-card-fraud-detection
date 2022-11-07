#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import sys
import pickle
import glob

import numpy as np
import pandas as pd

import datetime
import time

import random

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

from tqdm import tqdm
tqdm.pandas()


# In[ ]:


def generate_customer_profiles_table(n_customers, random_state=0):
    
    np.random.seed(random_state)
        
    customer_id_properties=[]
    
    # Generate customer properties from random distributions 
    for customer_id in range(n_customers):
        
        x_customer_id = np.random.uniform(0,100)
        y_customer_id = np.random.uniform(0,100)
        
        mean_amount = np.random.uniform(5,100) # Arbitrary (but sensible) value 
        std_amount = mean_amount/2 # Arbitrary (but sensible) value
        
        mean_nb_tx_per_day = np.random.uniform(0,4) # Arbitrary (but sensible) value 
        
        customer_id_properties.append([customer_id,
                                      x_customer_id, y_customer_id,
                                      mean_amount, std_amount,
                                      mean_nb_tx_per_day])
        
    customer_profiles_table = pd.DataFrame(customer_id_properties, columns=['CUSTOMER_ID',
                                                                      'x_customer_id', 'y_customer_id',
                                                                      'mean_amount', 'std_amount',
                                                                      'mean_nb_tx_per_day'])
    
    return customer_profiles_table


# In[ ]:


def generate_terminal_profiles_table(n_terminals, random_state=0):
    
    np.random.seed(random_state)
        
    terminal_id_properties=[]
    
    # Generate terminal properties from random distributions 
    for terminal_id in range(n_terminals):
        
        x_terminal_id = np.random.uniform(0,100)
        y_terminal_id = np.random.uniform(0,100)
        
        terminal_id_properties.append([terminal_id,
                                      x_terminal_id, y_terminal_id])
                                       
    terminal_profiles_table = pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID',
                                                                      'x_terminal_id', 'y_terminal_id'])
    
    return terminal_profiles_table


# In[ ]:


def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    
    # Use numpy arrays in the following to speed up computations
    
    # Location (x,y) of customer as numpy array
    x_y_customer = customer_profile[['x_customer_id','y_customer_id']].values.astype(float)
    
    # Squared difference in coordinates between customer and terminal locations
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    
    # Sum along rows and compute suared root to get distance
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
    
    # Get the indices of terminals which are at a distance less than r
    available_terminals = list(np.where(dist_x_y<r)[0])
    
    # Return the list of terminal IDs
    return available_terminals


# In[ ]:


def generate_transactions_table(customer_profile, start_date, day=0):
    
    customer_transactions = []
    
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))
        
    # Random number of transactions for that day 
    nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        
    # If nb_tx positive, let us generate transactions
    if nb_tx>0:
            
        for tx in range(nb_tx):
                
            # Time of transaction: Around noon, std 20000 seconds. This choice aims at simulating the fact that 
            # most transactions occur during the day.
            time_tx = int(np.random.normal(86400/2, 20000))
                
            # If transaction time between 0 and 86400, let us keep it, otherwise, let us discard it
            if (time_tx>0) and (time_tx<86400):
                    
                # Amount is drawn from a normal distribution  
                amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    
                # If amount negative, draw from a uniform distribution
                if amount<0:
                    amount = np.random.uniform(0,customer_profile.mean_amount*2)
                    
                amount=np.round(amount,decimals=2)
                    
                if len(customer_profile.available_terminals)>0:
                        
                    terminal_id = random.choice(customer_profile.available_terminals)
                    
                    customer_transactions.append([time_tx, day,
                                                  customer_profile.CUSTOMER_ID,
                                                  terminal_id, amount])
            
    customer_transactions = pd.DataFrame(customer_transactions, columns=['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'])
    
    if len(customer_transactions)>0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date)
        customer_transactions=customer_transactions[['TX_DATETIME','CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT','TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    
    return customer_transactions


# In[ ]:


def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df):
    
    # By default, all transactions are genuine
    transactions_df['TX_FRAUD']=0
    transactions_df['TX_FRAUD_SCENARIO']=0
    
    # Scenario 1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD']=1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD_SCENARIO']=1
    nb_frauds_scenario_1=transactions_df.TX_FRAUD.sum()
    print("Number of frauds from scenario 1: "+str(nb_frauds_scenario_1))
    
    # Scenario 2
        
    compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=1, random_state=np.random.randint(1, 1000))
        
    compromised_transactions=transactions_df[(transactions_df.TERMINAL_ID.isin(compromised_terminals))]
                            
    transactions_df.loc[compromised_transactions.index,'TX_FRAUD']=1
    transactions_df.loc[compromised_transactions.index,'TX_FRAUD_SCENARIO']=2
    
    nb_frauds_scenario_2=transactions_df.TX_FRAUD.sum()-nb_frauds_scenario_1
    print("Number of frauds from scenario 2: "+str(nb_frauds_scenario_2))
    
    # Scenario 3
        
    compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=10, random_state=np.random.randint(1, 1000)).values
          
    compromised_transactions=transactions_df[(transactions_df.CUSTOMER_ID.isin(compromised_customers))]
        
    nb_compromised_transactions=len(compromised_transactions)
        
        
    random.seed(np.random.randint(1, 1000))
    index_fauds = random.sample(list(compromised_transactions.index.values),k=int(nb_compromised_transactions/2))
        
    transactions_df.loc[index_fauds,'TX_AMOUNT']=transactions_df.loc[index_fauds,'TX_AMOUNT']*5
    transactions_df.loc[index_fauds,'TX_FRAUD']=1
    transactions_df.loc[index_fauds,'TX_FRAUD_SCENARIO']=3
        
                             
    nb_frauds_scenario_3=transactions_df.TX_FRAUD.sum()-nb_frauds_scenario_2-nb_frauds_scenario_1
    print("Number of frauds from scenario 3: "+str(nb_frauds_scenario_3))
    
    return transactions_df


# In[ ]:


def generate_dataset(customer_profiles_table, terminal_profiles_table, start_date="2018-01-01", r=5, start_index = 0):
    
    start_time=time.time()
    
    x_y_terminals = terminal_profiles_table[['x_terminal_id','y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    # With Pandarallel
    #customer_profiles_table['available_terminals'] = customer_profiles_table.parallel_apply(lambda x : get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1)
    customer_profiles_table['nb_terminals']=customer_profiles_table.available_terminals.apply(len)
    print("Time to associate terminals to customers: {0:.2}s".format(time.time()-start_time))
    
    start_time=time.time()
    
    transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').apply(lambda x : generate_transactions_table(x.iloc[0], start_date)).reset_index(drop=True)
    # With Pandarallel
    #transactions_df=customer_profiles_table.groupby('CUSTOMER_ID').parallel_apply(lambda x : generate_transactions_table(x.iloc[0], start_date)).reset_index(drop=True)
    print("Time to generate transactions: {0:.2}s".format(time.time()-start_time))
    
    # Sort transactions chronologically
    transactions_df=transactions_df.sort_values('TX_DATETIME')
    # Reset indices, starting from 0
    transactions_df.reset_index(inplace=True,drop=True)
    transactions_df.reset_index(inplace=True)
    # TRANSACTION_ID are the dataframe indices, starting from 0
    transactions_df.rename(columns = {'index':'TRANSACTION_ID'}, inplace = True)
    transactions_df.TRANSACTION_ID += start_index

    transactions_df = add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df)

    return transactions_df


# In[ ]:


def seconds_until_midnight(last):
    
    tomorrow = last + datetime.timedelta(1)
    midnight = datetime.datetime(year = tomorrow.year, month = tomorrow.month, day = tomorrow.day,
                                hour = 0, minute = 0, second = 0)
    return (midnight - last).seconds


# In[ ]:


DIR_OUTPUT = "/home/ubuntu/Fraud-detection/simulated-data-raw/" #"simulated-data-raw/"

if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)


# In[ ]:


#get the latest slice of historical data

list_of_files = glob.glob(DIR_OUTPUT+'*')
latest_file = max(list_of_files, key = os.path.getctime)

with open(r"{}".format(latest_file), "rb") as input_file:
    latest_data = pickle.load(input_file)


# In[ ]:


#today's date is the day after the latest historical slice

today = latest_data.iloc[-1, :]['TX_DATETIME'] + datetime.timedelta(days=1)


# In[ ]:


N_CUSTOMERS = 10000
N_TERMINALS = 1000

customer_profiles_table = generate_customer_profiles_table(n_customers = N_CUSTOMERS, random_state = np.random.randint(1, 1000))
terminal_profiles_table = generate_terminal_profiles_table(n_terminals = N_TERMINALS, random_state = np.random.randint(1, 1000))

todays_date = f'{today.year}-{today.month}-{today.day}'

transactions_df = generate_dataset(customer_profiles_table,
                                   terminal_profiles_table,
                                   start_date = todays_date,
                                   r=5,
                                   start_index = 0)


print(transactions_df.shape)

print(transactions_df.head(3))

daily_transactions = transactions_df.sort_values('TX_TIME_SECONDS')

#the following features start at the values of the latest historical slice
daily_transactions.index += latest_data.index[-1]+1
daily_transactions['TRANSACTION_ID'] += latest_data.iloc[-1, :]['TRANSACTION_ID']+1
daily_transactions['TX_TIME_SECONDS'] += latest_data.iloc[-1, :]['TX_TIME_SECONDS']+seconds_until_midnight(latest_data.iloc[-1, :]['TX_DATETIME'])
daily_transactions['TX_TIME_DAYS'] += latest_data.iloc[-1, :]['TX_TIME_DAYS']+1

daily_transactions = daily_transactions[latest_data.columns]


# In[ ]:


filename_output = todays_date + '.pkl'

daily_transactions.to_pickle(DIR_OUTPUT+filename_output)


# In[ ]:


get_ipython().system("sudo -u hdfs hdfs dfs -put '/home/ubuntu/Fraud-detection/simulated-data-raw/{filename_output}' /data")


# In[ ]:


print('Daily batch successfully loaded to hdfs')

