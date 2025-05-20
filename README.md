

# Sentiment Analysis with Spark Streaming and Kafka Server

This project demonstrates a real-time sentiment analysis training and evaluating pipeline for VSFC dataset using Apache Spark Streaming and Kafka. The architecture consists of two main components:

* **Producer**: Streaming batch using **Kafka Procedure**.

* **Consumer**: Using spark streaming connecting with kafka topic to get batch from **Producer**

* **Main**: Have two mode training and tessting.

## 🧾 Dataset
[*Dataset link*](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)


## 📦 Prerequisites

* Python 3.10
* Java 17.0.12
* Apache Spark 3.4.4
* Kafka 2.12-3.6.0

## 🚀 Usage
```
!git clone https://github.com/MrNquyen/Spark-Streaming-with-Kafka.git
```

### 1. Start with kafka

```
# Start server 
bin\windows\zookeeper-server-start.bat config\zookeeper.properties

# Start Cluster
bin\windows\kafka-server-start.bat config\server.properties
```

### 2. Sending batch using kafka

```
# Stream train data
python producer.py --config config/config.yml --split train

# Stream test data
python producer.py --config config/config.yml --split test
```

* Topic: *VSFC_Sentiment_train* is used for training
* Topic: *VSFC_Sentiment_test* is used for testing


### 3. Training and Testing mode
Finetuning pretrained model on UIT-VSFC dataset

```
# Training
python main.py --config config/config.yml --mode train

# Testing
python main.py --config config/config.yml --mode test
```

## 📈 Evaluation
* Inspect console logs for per-batch accuracy, precision, recall, and F1-score.
* After finishing prediction mode, review final aggregate metrics.

