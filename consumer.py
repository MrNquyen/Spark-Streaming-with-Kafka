import findspark
findspark.init()

import pyspark
import time as time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, IntegerType 
from pyspark.sql.functions import from_json,current_timestamp
from pyspark.ml.feature import VectorAssembler
from IPython.display import display, clear_output

#------------------------GLOBAL--------------------------------------------------------------
kafka_server = 'localhost:9092'
topic_name = 'VSFC_Sentiment'
scala_version = '2.12'  # your scala version
spark_version = '3.4.4'  # your spark version
startingOffsets = 'earliest'

packages = [
    f'org.apache.spark:spark-sql-kafka-0-10_{scala_version}:{spark_version}',
    'org.apache.kafka:kafka-clients:2.8.0'  # your kafka version
]


#------------------------FUNCTION--------------------------------------------------------------
# Define schema
schema = StructType([
    StructField("sentences", ArrayType(StringType()), True),
    StructField("labels", ArrayType(StringType()), True),
    StructField("label_ids", ArrayType(IntegerType()), True),
])


#------------------------SPARK_SESSION--------------------------------------------------------------
# Initialize Spark session
checkpoint_kafka_path = Path("tmp/checkpoints")
if not checkpoint_kafka_path.exists():
    checkpoint_kafka_path.mkdir(parents=True, exist_ok=True)

spark = SparkSession.builder \
    .master("local") \
    .appName("VSFC_Sentiment_Analysis") \
    .config("checkpointLocation", "tmp/checkpoints") \
    .config("spark.jars.packages", ",".join(packages)) \
    .getOrCreate()

#------------------------READ_STREM--------------------------------------------------------------
# startingOffsets: "earliest" - Load everything from the begining of the topic
# startingOffsets: "latest" - only new messages arriving after streaming starts

def load_stream_df(kafka_server, topic_name):
    stream_df = spark \
        .readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_server) \
        .option("subscribe", topic_name) \
        .option("startingOffsets", startingOffsets) \
        .load()
    return stream_df

def parsing_stream_df(df):
    json_df = df.selectExpr("CAST(key AS STRING)", "CAST(value AS STRING) as value", "CAST(offset as BIGINT) as offset")
    json_expanded_df = json_df \
        .withColumn("value", from_json(json_df["value"], schema)) \
        .withColumn("offset", col("offset").cast("long")) \
        .select("value.*", "offset")
    exploded_df = json_expanded_df.select("sentences", "labels", "label_ids", "offset")
    return exploded_df

def query_stop(query):
    query.stop()


#------------------------DATALOADER--------------------------------------------------------------
class DataLoaderStreamKafka:
    def __init__(self, kafka_server, topic_name, query_name="temp_view"):
        stream_df = load_stream_df(kafka_server, topic_name)
        parsed_stream_df = parsing_stream_df(stream_df)
        self.data = []
        self.current_idx = 0
        self.query_name = query_name
        self.query = self.query_start(
            parsed_df=parsed_stream_df,
            query_name=self.query_name
        )


    def query_start(self, parsed_df, query_name="temp_view"):
        self.query = parsed_df.writeStream \
            .format("memory") \
            .queryName(query_name) \
            .start()


    def get_current_length(self):
        return spark.sql(f"SELECT COUNT(*) FROM {self.query_name}").collect()[0][0]
            

    def display_results(self, result):
        display(result)
        display(f"Last update: {time.strftime('%H:%M:%S')}")
        clear_output(wait=True)


    def get_dataloader(self, num_batches):
        for idx in range(num_batches):
            if idx < len(self.data):
                yield self.data[idx]
            else:
                yield self.get_batches(idx)


    def get_batches(self, idx):
        # Idx out of range
        last_current_idx = self.current_idx 
        self.current_idx = idx
        while idx >= self.get_current_length():

            clear_output(wait=True)
            display("Wait for new records")
            display(f"Number of records: {self.get_current_length()}")
            
            time.sleep(2)
        while True:
            try:
                # Load idx from current records
                result = spark.sql(f"""
                    SELECT * FROM {self.query_name}
                    WHERE offset >= {idx}
                    ORDER BY offset ASC
                    LIMIT 10
                """)
                self.display_results(result=result.toPandas())
                value_dict = {
                    "sentences": result.collect()[0]['sentences'],
                    "labels": result.collect()[0]['labels'],
                    "label_ids": result.collect()[0]['label_ids'],
                }
                self.data.append(value_dict)
                return value_dict
            except Exception as e:
                self.current_idx = last_current_idx
                display(f"Error during display: {str(e)}")
                time.sleep(5)

        

