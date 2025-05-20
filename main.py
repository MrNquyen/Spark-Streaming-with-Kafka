#------------------------GLOBAL--------------------------------------------------------------
import argparse
from utils import *
from consumer import DataLoaderStreamKafka
from model import SentimentBertModel

#------------------------GLOBAL--------------------------------------------------------------
kafka_server = 'localhost:9092'
topic_name = 'VSFC_Sentiment_{split}'

#------------------------FUNCTION--------------------------------------------------------------
def getargs():
    parser = argparse.ArgumentParser(description="Streams a file to a Spark Streaming Context")
    parser.add_argument("--config", "-c", help="Path to config file", required=True, type=str)
    parser.add_argument("--split", "-spl", help="Which split want to use", required=True, type=str, default="train", choices=["train", "test"])
    parser.add_argument("--save_dir", help="Your save directory", type=str, default="save")
    return parser.parse_args()


if __name__=="__main__":
    #--- Load Params
    args = getargs() 
    config = load_yml(args.config)
    topic_name = topic_name.format(split=args.split)

    #--- Define Loader
    print("Load Dataloader")
    stream_loader = DataLoaderStreamKafka(
        kafka_server=kafka_server,
        topic_name=topic_name
    )

    #--- Define Model

    
        

    #--- Select mode
    if args.split=="train":
        #---Training
        print("Load Model for Training")
        sentiment_model = SentimentBertModel(config=config)
        sentiment_model.train(stream_loader)
    
    elif args.split=="test":
        #---Evaluating
        print("Load Model for Evaluate")
        config["model"]["model_name"] = f"{args.save_dir}/sentiment_bert"
        sentiment_model = SentimentBertModel(config=config)
        sentiment_model.eval(stream_loader, save_dir=args.save_dir)
