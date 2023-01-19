import findspark
findspark.init()
findspark.find()

from IPython import get_ipython

import os
import sys
import pickle

import numpy as np
import pandas as pd

import datetime
import time

import random

from tqdm import tqdm
tqdm.pandas()

import warnings
warnings.filterwarnings("ignore")

from collections import Counter

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import dayofweek, dayofmonth, col, when, hour
from pyspark.sql.window import Window
from pyspark.sql import functions as F

from joblibspark import register_spark

def get_customer_spending_behaviour_features(spark_df, windows_size_in_days = [1, 7, 30]):
    
    days = lambda i: i * 86400
    
    for window_size in windows_size_in_days:
        
        windowSpec = Window().partitionBy(['CUSTOMER_ID']).orderBy(col('TX_DATETIME').cast('timestamp').cast('long')).rangeBetween(-days(window_size), 0)
        spark_df = spark_df.withColumn('CUSTOMER_ID_NB_TX_'+str(window_size)+'DAY_WINDOW', F.count('*').over(windowSpec)).orderBy('TX_DATETIME')
        spark_df = spark_df.withColumn('CUSTOMER_ID_AVG_AMOUNT_'+str(window_size)+'DAY_WINDOW', F.mean('TX_AMOUNT').over(windowSpec)).orderBy('TX_DATETIME')

    return spark_df

def get_count_risk_rolling_window(spark_df, delay_window = 7, windows_size_in_days = [1, 7, 30]):

    days = lambda i: i * 86400
    delay_period = days(delay_window)

    windowSpec = Window().partitionBy(['TERMINAL_ID']).orderBy(col('TX_DATETIME').cast('timestamp').cast('long')).rangeBetween(-delay_period, 0)
    spark_df = spark_df.withColumn('NB_FRAUD_DELAY', F.sum('TX_FRAUD').over(windowSpec)).orderBy('TX_DATETIME')
    spark_df = spark_df.withColumn('NB_TX_DELAY', F.count('TX_FRAUD').over(windowSpec)).orderBy('TX_DATETIME')

    for window_size in windows_size_in_days:

        windowSpec = Window().partitionBy(['TERMINAL_ID']).orderBy(col('TX_DATETIME').cast('timestamp').cast('long')).rangeBetween(-days(window_size)-delay_period, 0)
        spark_df = spark_df.withColumn('NB_FRAUD_DELAY_WINDOW', F.sum('TX_FRAUD').over(windowSpec)).orderBy('TX_DATETIME')
        spark_df = spark_df.withColumn('NB_TX_DELAY_WINDOW', F.count('TX_FRAUD').over(windowSpec)).orderBy('TX_DATETIME')
        spark_df = spark_df.withColumn('NB_FRAUD_WINDOW', spark_df['NB_FRAUD_DELAY_WINDOW']-spark_df['NB_FRAUD_DELAY'])

        spark_df = spark_df.withColumn('TERMINAL_ID_NB_TX_'+str(window_size)+'DAY_WINDOW', spark_df['NB_TX_DELAY_WINDOW']-spark_df['NB_TX_DELAY'])
        spark_df = spark_df.withColumn('TERMINAL_ID_RISK_'+str(window_size)+'DAY_WINDOW', spark_df['NB_FRAUD_WINDOW']/spark_df['TERMINAL_ID_NB_TX_'+str(window_size)+'DAY_WINDOW'])

    spark_df = spark_df.na.fill(0)
    spark_df = spark_df.drop('NB_FRAUD_DELAY', 'NB_TX_DELAY', 'NB_FRAUD_DELAY_WINDOW', 'NB_TX_DELAY_WINDOW', 'NB_FRAUD_WINDOW')

    return spark_df

def preprocess_transactions(transactions_df):

    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    spark_df = spark.createDataFrame(transactions_df)

    spark_df = spark_df.withColumn('TX_DURING_WEEKEND', dayofweek(spark_df.TX_DATETIME))
    spark_df = spark_df.withColumn('TX_DURING_WEEKEND', when((spark_df['TX_DURING_WEEKEND'] == 1) | (spark_df['TX_DURING_WEEKEND'] == 7), 1).otherwise(0))

    spark_df = spark_df.withColumn('TX_DURING_NIGHT', hour(spark_df.TX_DATETIME))
    spark_df = spark_df.withColumn('TX_DURING_NIGHT', when(spark_df['TX_DURING_NIGHT'] <= 6, 1).otherwise(0))

    spark_df = get_customer_spending_behaviour_features(spark_df)
    spark_df = get_count_risk_rolling_window(spark_df)
    
    return spark_df

def save_processed(transactions_df, DIR_OUTPUT):
    
    transactions_df = transactions_df.toPandas()

    start_date = transactions_df.loc[0, 'TX_DATETIME']
    start_date = datetime.datetime.strptime(f"{start_date.year}-{start_date.month}-{start_date.day}", "%Y-%m-%d")

    for day in range(transactions_df.TX_TIME_DAYS.max() - transactions_df.TX_TIME_DAYS.min()+1):
        
        transactions_day = transactions_df[transactions_df.TX_TIME_DAYS==transactions_df.TX_TIME_DAYS.min()+day].sort_values('TX_TIME_SECONDS')

        date = start_date + datetime.timedelta(days=day)
        filename_output = date.strftime("%Y-%m-%d")+'.pkl'

        # Protocol=4 required for Google Colab
        transactions_day.to_pickle(DIR_OUTPUT+filename_output, protocol=4)

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

def get_preprocessed_data(days, INPUT_DIR, OUTPUT_DIR):
    
    transactions_df = read_from_files(INPUT_DIR)
    start_date = transactions_df.iloc[-1, :]['TX_DATETIME'] - datetime.timedelta(days=days)
    transactions_df = transactions_df[transactions_df.TX_DATETIME > start_date]
    transactions_df = preprocess_transactions(transactions_df)
    save_processed(transactions_df, OUTPUT_DIR)

days = 90
INPUT_DIR = '/home/ubuntu/Fraud-detection/simulated-data-raw'
OUTPUT_DIR = '/home/ubuntu/Fraud-detection/simulated-data-processed/'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

get_preprocessed_data(days, INPUT_DIR, OUTPUT_DIR)
get_ipython().system('sudo -u hdfs hdfs dfs -put /home/ubuntu/Fraud-detection/simulated-data-processed/* /data/simulated-data-processed/')