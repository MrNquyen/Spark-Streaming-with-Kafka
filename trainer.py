import json
import os
import transformers
import sparknlp
import argparse

from IPython.display import display, clear_output
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sparknlp.base import DocumentAssembler, Pipeline


from model import SentimentBertModel
from consumer import DataLoaderStreamKafka
from utils import *

# Start spark nlp
# display("Start spark-nlp")
# sparknlp.start()
# clear_output(wait=True)

#------------------------GLOBAL--------------------------------------------------------------
kafka_server = 'localhost:9092'
topic_name = 'VSFC_Sentiment'
scala_version = '2.12'  # your scala version
spark_version = '3.4.4'  # your spark version
startingOffsets = 'lastest'

#------------------------FUNCTION--------------------------------------------------------------
def getargs():
    parser = argparse.ArgumentParser(description="Streams a file to a Spark Streaming Context")
    parser.add_argument("--config", "-c", help="Path to config file", required=True, type=str)
    return parser.parse_args()

#------------------------TRAINER--------------------------------------------------------------
class Trainer:
    def __init__(self, config):
        self.config = config
        self.model, self.tokenizer = self.init_model()


    def init_model(self):
        model_name = self.config['training']['model']
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
#------------------------MAIN--------------------------------------------------------------
if __name__=="__main__":
    #--- Load Params
    args = getargs() 
    config = load_yml(args.config)

    #--- Define Loader
    print("Load Dataloader")
    stream_loader = DataLoaderStreamKafka(
        kafka_server=kafka_server,
        topic_name=topic_name
    )

    #--- Define Model
    print("Load Model")
    sentiment_model = SentimentBertModel(config=config)
    
    #---Training
    sentiment_model.train(stream_loader)