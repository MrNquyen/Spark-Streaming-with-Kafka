from kafka import KafkaProducer
from json import dumps
from time import sleep
from tqdm import tqdm
from torch.utils.data import DataLoader
from icecream import ic

import numpy as np
import pandas as pd
import argparse

from utils import load_yml
from vsfc import VSFCDataset, collate_fn

#------------------------GLOBAL--------------------------------------------------------------
kafka_server = 'localhost:9092'
topic_name = 'VSFC_Sentiment_{split}'

#------------------------FUNCTION--------------------------------------------------------------
def getargs():
    parser = argparse.ArgumentParser(description="Streams a file to a Spark Streaming Context")
    parser.add_argument("--config", "-c", help="Path to config file", required=True, type=str)
    parser.add_argument("--split", "-spl", help="Which split want to stream", required=True, type=str, default="train", choices=["train", "val", "test"])
    return parser.parse_args()

def setup_kafka_producer():
    producer = KafkaProducer(bootstrap_servers=kafka_server,value_serializer = lambda x:dumps(x).encode('utf-8'))
    return producer

#------------------------MAIN--------------------------------------------------------------
if __name__=="__main__":
    args = getargs() 
    config = load_yml(args.config)
    producer = setup_kafka_producer()

    #--- Set topic name
    topic_name = topic_name.format(split=args.split)

    #--- Load dataset
    dataloader = None
    if args.split=="train":
        dataset = VSFCDataset(config=config, split="train")
        dataloader = DataLoader(dataset, shuffle=False, batch_size=config["training"]["batch_size"], collate_fn=collate_fn)
    elif args.split=="val":
        dataset = VSFCDataset(config=config, split="val")
        dataloader = DataLoader(dataset, shuffle=True, batch_size=config["training"]["batch_size"], collate_fn=collate_fn)
    elif args.split=="test":
        dataset = VSFCDataset(config=config, split="test")
        dataloader = DataLoader(dataset, shuffle=True, batch_size=config["training"]["batch_size"], collate_fn=collate_fn)
    else:
        raise Exception("No split found")


    #--- Iterate and post batch to spark
    for id, (sentences, labels, label_ids) in enumerate(dataloader):
        batch_dict = {
            "sentences": sentences,
            "labels": labels,
            "label_ids": label_ids,
        }
        print(f"Batch {id} sent")
        producer.send(topic_name, value=batch_dict)
        sleep(3)