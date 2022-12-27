#!/usr/bin/env python

import os
import findspark

findspark.init()
findspark.find()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["MLFLOW_TRACKING_URI"] = "http://84.252.138.16:8000/"

import json
import argparse
from typing import Dict, NamedTuple
from collections import namedtuple
import logging

import kafka
from kafka import KafkaConsumer, TopicPartition

import sys
import pickle
import glob

import numpy as np
import pandas as pd

import datetime
import time

import yaml

import random

import pyspark
from pyspark.ml import PipelineModel
import mlflow

from preprocess_new_transaction import preprocess_transaction

DIR_PROC = "/home/ubuntu/Fraud-detection/simulated-data-processed/"

customer_columns = ['CUSTOMER_ID', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
                        'CUSTOMER_ID_NB_TX_7DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
                        'CUSTOMER_ID_NB_TX_30DAY_WINDOW', 'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW']
    
terminal_columns = ['TERMINAL_ID', 'TERMINAL_ID_NB_TX_1DAY_WINDOW', 'TERMINAL_ID_RISK_1DAY_WINDOW',
                        'TERMINAL_ID_NB_TX_7DAY_WINDOW', 'TERMINAL_ID_RISK_7DAY_WINDOW',
                        'TERMINAL_ID_NB_TX_30DAY_WINDOW', 'TERMINAL_ID_RISK_30DAY_WINDOW']


processed = []

model_name = 'RF_classifier'

#prod_model = mlflow.spark.load_model(f'models:/{model_name}/Production')
prod_model = PipelineModel.load('/tmp/mlflow/df591540-01dd-44b6-9fd4-ad63e395e65a')

class RecordMetadata(NamedTuple):
    topic: str
    partition: int
    offset: int


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "-g", "--group_id", required=True, help="kafka consumer group_id"
    )
    argparser.add_argument(
        "-b",
        "--bootstrap_server",
        default="rc1b-tdjco20jo3dnkt2m.mdb.yandexcloud.net:9091",
        help="kafka server address:port",
    )
    argparser.add_argument(
        "-u", "--user", required=True, default="user", help="kafka user"
    )
    argparser.add_argument(
        "-p", "--password", required=True, default="password", help="kafka user password"
    )
    argparser.add_argument(
        "-tc", "--topicconsumer", default="raw-data", help="kafka topic to consume raw data"
    )

    argparser.add_argument(
        "-tp", "--topicproducer", default="predictions", help="kafka topic to produce predictions"
    )
    
    argparser.add_argument(
        "-n", "--nmessages",
        default=10,
        type=int,
        help="number of messages to process",
    )

    args = argparser.parse_args()

    consumer = KafkaConsumer(
        bootstrap_servers=args.bootstrap_server,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username=args.user,
        sasl_plain_password=args.password,
        ssl_cafile="YandexCA.crt",
        group_id=args.group_id,
        value_deserializer=json.loads,
    )

    producer = kafka.KafkaProducer(
        bootstrap_servers=args.bootstrap_server,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username=args.user,
        sasl_plain_password=args.password,
        ssl_cafile="YandexCA.crt",
        value_serializer=serialize,
    )

    consumer.subscribe(topics=[args.topicconsumer])

    global DIR_PROC
    global customer_columns
    global terminal_columns
    global processed
    global prod_model

    preprocessed_data = generate_preprocessed(consumer, args.nmessages)
    preprocessed_data = pd.concat(preprocessed_data)
    print('Data preprocessed correctly')
    
    spark = pyspark.sql.SparkSession.builder.getOrCreate()
    
    data = spark.createDataFrame(preprocessed_data)
    
    print('Predicting outcomes...')
    predictions = prod_model.transform(data)
    results_df = predictions.select(['TRANSACTION_ID', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'prediction', 'probability']).toPandas()
    
    print('Inference complete! Sending results to "predictions" topic using Kafka Producer...')
    
    try:
        for i in range(results_df.shape[0]-1):
            pred = results_df.iloc[i, :].to_dict()
            record_md = send_message(producer, args.topicproducer, pred)
            print(
                f"Msg sent. Topic: {record_md.topic}, partition:{record_md.partition}, offset:{record_md.offset}"
            )
    except kafka.errors.KafkaError as err:
        logging.exception(err)
    producer.flush()
    producer.close()

def generate_preprocessed(consumer, n):
    print("Waiting for raw data...")
    
    i = -1
    
#     tp = TopicPartition(topic, 0)
#     lastOffset = consumer.end_offsets([tp])[tp]
    
    for msg in consumer:
        
        tr = yaml.load(msg.value, Loader=yaml.Loader)
        value = preprocess_transaction(tr, DIR_PROC, customer_columns, terminal_columns)
        processed.append(value)
        i+=1
        if i == n:
            break
    return processed

def send_message(producer: kafka.KafkaProducer, topic: str, pred: dict) -> RecordMetadata:

    future = producer.send(
        topic=topic,
        key=str(pred['TRANSACTION_ID']).encode("ascii"),
        value=str(pred),
    )

    # Block for 'synchronous' sends
    record_metadata = future.get(timeout=1)
    return RecordMetadata(
        topic=record_metadata.topic,
        partition=record_metadata.partition,
        offset=record_metadata.offset,
    )


def serialize(msg: Dict) -> bytes:
    return json.dumps(msg).encode("utf-8")


if __name__ == "__main__":
    main()
