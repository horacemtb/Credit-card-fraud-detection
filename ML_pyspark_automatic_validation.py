#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pyspark


# In[2]:


#!pip install joblibspark


# In[3]:


#!pip install findspark


# In[4]:


#!pip install shap


# In[5]:


import findspark


# In[6]:


findspark.init()
findspark.find()


# In[7]:


import os


# In[8]:


os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["MLFLOW_TRACKING_URI"] = "http://158.160.13.60:8000/"


# In[9]:


import sys
import pickle

import numpy as np
import pandas as pd

import datetime
import time

import random

from tqdm import tqdm
tqdm.pandas()

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', '')

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from scipy.stats import ttest_ind

from collections import Counter

import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.ensemble import RandomForestClassifier
from pyspark.ml.classification import RandomForestClassifier as pysparkRF
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml import Pipeline

from joblibspark import register_spark
from sklearn.utils import parallel_backend

import mlflow

from preprocess import get_preprocessed_data


# In[10]:


import shap


# In[11]:


INPUT_DIR = '/home/ubuntu/Fraud-detection/simulated-data-raw'
OUTPUT_DIR = '/home/ubuntu/Fraud-detection/simulated-data-processed/'
DAYS = 90


# In[12]:


if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# In[13]:


get_preprocessed_data(DAYS, INPUT_DIR, OUTPUT_DIR)


# In[14]:


INPUT_DIR_PROC = '/home/ubuntu/Fraud-detection/simulated-data-processed'


# In[15]:


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
    df_final=df_final.replace([-1], 0)
    
    return df_final


# In[16]:


print('Loading data...')
transactions_df = read_from_files(INPUT_DIR_PROC)
#transactions_df = pd.read_csv('transactions_df.csv')


# In[17]:


transactions_df['TX_DATETIME'] = pd.to_datetime(transactions_df['TX_DATETIME'])


# In[18]:


start_training_date = transactions_df.iloc[-1, :]['TX_DATETIME'] - datetime.timedelta(days=49)


# In[19]:


print(f'Start date: {start_training_date.year}-{start_training_date.month}-{start_training_date.day}')


# In[20]:


def get_train_val_set(transactions_df,
                       start_date_training,
                       delta_train=28, delta_delay=7, delta_val=7):
    
    # Get the training set data
    train_df = transactions_df[(transactions_df.TX_DATETIME>=start_date_training) &
                               (transactions_df.TX_DATETIME<start_date_training+datetime.timedelta(days=delta_train))]
    
    # Get the test set data
    val_df = []
    
    # Note: Cards known to be compromised after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day-delay_period) are removed
    
    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD==1].CUSTOMER_ID)
    
    # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()
    
    # Then, for each day of the test set
    for day in range(delta_val):
    
        # Get test data for that day
        val_df_day = transactions_df[transactions_df.TX_TIME_DAYS==start_tx_time_days_training+
                                                                    delta_train+delta_delay+
                                                                    day]
        
        # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded customers
        val_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS==start_tx_time_days_training+
                                                                                delta_train+
                                                                                day-1]
        
        new_defrauded_customers = set(val_df_day_delay_period[val_df_day_delay_period.TX_FRAUD==1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)
        
        val_df_day = val_df_day[~val_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]
        
        val_df.append(val_df_day)
        
    val_df = pd.concat(val_df)
    
    # Sort data sets by ascending order of transaction ID
    train_df=train_df.sort_values('TRANSACTION_ID')
    val_df=val_df.sort_values('TRANSACTION_ID')
    
    return (train_df, val_df)


# In[21]:


(train_df, val_df) = get_train_val_set(transactions_df, start_training_date,
                                         delta_train=28, delta_delay=7, delta_val=7)


# In[22]:


print('Train shape: ', train_df.shape)
print('Val shape: ', val_df.shape)
print('Train frauds: ', train_df[train_df.TX_FRAUD==1].shape[0])
print('Val frauds: ', val_df[val_df.TX_FRAUD==1].shape[0])
print('Val fraud percentage', val_df[val_df.TX_FRAUD==1].shape[0]/val_df.shape[0]*100)


# In[23]:


full_df = pd.concat([train_df, val_df], axis=0)


# In[24]:


split_index = [-1 if x in train_df.index else 0 for x in full_df.index]


# In[25]:


pds = PredefinedSplit(test_fold = split_index)


# In[26]:


output_feature = "TX_FRAUD"

input_features = ['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                  'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                  'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                  'TERMINAL_ID_RISK_30DAY_WINDOW']


# In[27]:


X = full_df[input_features]
y = full_df[output_feature]


# In[28]:


estimator = RandomForestClassifier()

params_space = {'n_estimators': [50, 400, 600, 800],
                'max_features': ['auto', 'sqrt'],
                'max_depth': [1, 5, 8],
                'criterion': ['gini', 'entropy'], 
                'min_samples_leaf': [1, 3, 5]
                }


# In[29]:


print('Searching for the best params...')


# In[30]:


N_ITER = 10


# In[31]:


register_spark()


# In[32]:


parallelism = 8
with parallel_backend("spark", n_jobs=parallelism):
    grid_search = RandomizedSearchCV(estimator = estimator,
                                param_distributions = params_space,
                                n_iter = N_ITER,
                                scoring = 'roc_auc',
                                cv = pds,
                                verbose = 2
                                )
    grid_search.fit(X, y)


# In[33]:


pysparkRF_params = {'numTrees': grid_search.best_params_['n_estimators'],
                    'featureSubsetStrategy': grid_search.best_params_['max_features'],
                    'maxDepth': grid_search.best_params_['max_depth'],
                    'minInstancesPerNode': grid_search.best_params_['min_samples_leaf'], 
                    'impurity': grid_search.best_params_['criterion']}


# In[34]:


print(f'Best Random Forest params: {pysparkRF_params}')


# In[35]:


spark = pyspark.sql.SparkSession.builder.getOrCreate()

spark_train = spark.createDataFrame(train_df) 
spark_val = spark.createDataFrame(val_df) 


# In[36]:


mlflow.set_experiment('Predicting fraud')


# In[37]:


def train(params):

    mlflow.start_run()

    assembler = VectorAssembler(inputCols=input_features, outputCol='features')
    classifier = pysparkRF(**params, featuresCol = 'features', labelCol = 'TX_FRAUD')
    pipeline = Pipeline(stages = [assembler, classifier])

    model = pipeline.fit(dataset = spark_train)
    predictions = model.transform(spark_val)

    evaluator = BinaryClassificationEvaluator(rawPredictionCol = "rawPrediction", labelCol = "TX_FRAUD")
    results_df = predictions.select(['TX_FRAUD', 'prediction', 'probability']).toPandas()
    results_df['probability'] = results_df['probability'].apply(lambda x: x[1])

    roc_auc = round(evaluator.evaluate(predictions), 3)
    precision = round(metrics.precision_score(results_df.TX_FRAUD, results_df.prediction), 3)
    recall = round(metrics.recall_score(results_df.TX_FRAUD, results_df.prediction), 3)

    print(f'ROC AUC: {roc_auc}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')

    mlflow.log_metric('roc_auc', roc_auc)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('recall', recall)
    mlflow.spark.log_model(model, 'RF_pipeline')

    mlflow.end_run()
    
    return results_df


# In[38]:


print('Training...')


# In[39]:


results_df = train(pysparkRF_params)


# In[40]:


mlflow_run = mlflow.search_runs().iloc[0]


# In[41]:


print(mlflow_run)


# In[42]:


model_name = 'RF_classifier'


# In[43]:


new_model_version = mlflow.register_model(f'runs:/{mlflow_run.run_id}/RF_pipeline', model_name)


# In[44]:


# run this cell when serving your first production model

# client = mlflow.tracking.MlflowClient()
# client.transition_model_version_stage(
#   name = model_name,
#   version = new_model_version.version,
#   stage = "Production"
# )


# In[45]:


def get_prod_model(model_name):
    
    prod_model = mlflow.spark.load_model(f'models:/{model_name}/Production')
    return prod_model


# In[46]:


prod_model = get_prod_model(model_name)


# In[47]:


# explainer = shap.TreeExplainer(prod_model.stages[1], feature_names = input_features)
# shap_values = explainer.shap_values(spark_val.toPandas()[input_features].iloc[5879:6215, :])
# shap.initjs()
# shap.force_plot(explainer.expected_value[1], shap_values[1], feature_names=input_features)


# In[48]:


def validate(validation_data, prod_model, cand_model, cand_model_results_df, model_name, input_features,
             bootstrap_iterations = 100, alpha = 0.05):
    
    #use current production model to get predictions
    prod_model_predictions = prod_model.transform(validation_data)
    prod_model_results_df = prod_model_predictions.select(['TX_FRAUD', 'prediction', 'probability']).toPandas()
    prod_model_results_df['probability'] = prod_model_results_df['probability'].apply(lambda x: x[1])
    
    prod_scores = pd.DataFrame(data = {'ROC AUC': 0.0}, index=range(bootstrap_iterations))
    cand_scores = pd.DataFrame(data = {'ROC AUC': 0.0}, index=range(bootstrap_iterations))
    
    #use bootstrap to generate two distributions
    for i in range(bootstrap_iterations):
        prod_sample = prod_model_results_df.sample(frac=1.0, replace=True)
        cand_sample = cand_model_results_df.sample(frac=1.0, replace=True)
        prod_scores.loc[i, 'ROC AUC'] = metrics.roc_auc_score(prod_sample['TX_FRAUD'], prod_sample['probability'])
        cand_scores.loc[i, 'ROC AUC'] = metrics.roc_auc_score(cand_sample['TX_FRAUD'], cand_sample['probability'])
    
    #plot the two distributions
    ax = sns.histplot(x = prod_scores['ROC AUC'], alpha = 0.3)
    sns_plot = sns.histplot(x = cand_scores['ROC AUC'], alpha = 0.3, color = 'orange', ax = ax)
    plt.close()
    
    #p-test
    alpha = 0.05    
    pvalue = ttest_ind(prod_scores['ROC AUC'], cand_scores['ROC AUC']).pvalue
    
    if pvalue < alpha:
        print('Reject null hypothesis')
        if prod_scores['ROC AUC'].mean() > cand_scores['ROC AUC'].mean():
            print('Keep current prod model')
        else:
            print('Serve new model')
            client = mlflow.tracking.MlflowClient()
            client.transition_model_version_stage(name = model_name,
                                                  version = cand_model.version,
                                                  stage = "Production"
                                                 )
    else:
        print('Accept null hypothesis')
        print('Keep current prod model')
        
    final_model = mlflow.spark.load_model(f'models:/{model_name}/Production')
    
    random_sample = np.random.randint(low=0, high=validation_data.count()-1, size=1)[0]
    explainer = shap.TreeExplainer(final_model.stages[1], feature_names = input_features)
    shap_values = explainer.shap_values(validation_data.toPandas()[input_features].iloc[random_sample, :])
    shap_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], feature_names=input_features, show=False, matplotlib=True)

    mlflow.set_experiment('Report')

    mlflow.start_run()
    
    mlflow.log_metric('p-value', pvalue)
    mlflow.log_figure(sns_plot.figure, 'distribution_plot.png')
    mlflow.log_figure(shap_plot, 'shap_plot.png')
    
    mlflow.end_run()


# In[49]:


validate(spark_val, prod_model, new_model_version, results_df, model_name, input_features)


# In[50]:


spark.stop()


# In[51]:


get_ipython().system('rm /home/ubuntu/Fraud-detection/simulated-data-processed/*')

