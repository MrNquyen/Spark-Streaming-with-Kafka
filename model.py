import torch
import json
import pandas
import os

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pathlib import Path
from icecream import ic
from tqdm import tqdm
from typing import List
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import Adam
from transformers.optimization import get_linear_schedule_with_warmup
from consumer import DataLoaderStreamKafka


class SentimentBertModel:
    def __init__(self, config):
        model_name = config['model']['model_name']
        self.device = config['training']['device']
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = config
        self.model_save_dir = config["save"]


    def train(self, dataloader: DataLoaderStreamKafka):
        optimizer = Adam(self.model.parameters(), lr=5e-5)
        num_epochs = self.config['training']['epochs']
        num_training_steps = num_epochs * self.config['training']['batches_per_epoch']
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['training']['num_warmup_steps'],
            num_training_steps=num_training_steps
        )
        progress_bar = tqdm(range(num_training_steps))
        self.model.train()
        for epoch in range(num_epochs):
            for batch in dataloader.get_dataloader(self.config['training']['batches_per_epoch']):
                sentences = batch["sentences"]
                labels = batch["labels"]
                label_ids = batch["label_ids"]
                
                # Encode
                batch_info = self.encode_batch(sentences)
                batch_info["labels"] = torch.tensor(label_ids)
                # ic(batch_info)
                # ic(batch_info.keys())
                # ic([type(v) for v in batch_info.values()])
                batch_info = {k: v.to(self.device) for k, v in batch_info.items()}
                
                # Forward pass
                outputs = self.model(**batch_info)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                progress_bar.update(1)
                progress_bar.set_description(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        model_save_path = Path(os.path.join(self.model_save_dir, "sentiment_bert"))
        if not model_save_path.exists():
            model_save_path.mkdir(parents=True, exist_ok=True)
        self.save_model(save_path=model_save_path)


    def eval(self, dataloader: DataLoaderStreamKafka, save_dir: str):
        all_eval = {
            "gt": [],
            'pred': []
        }
        for batch in dataloader.get_dataloader(self.config['training']['batches_per_epoch']):
            sentences = batch["sentences"]
            labels = batch["labels"]
            label_ids = batch["label_ids"]

            batch_info = self.encode_batch(sentences)
            batch_info["labels"] = torch.tensor(label_ids)
            batch_info = {k: v.to(self.device) for k, v in batch_info.items()}
            
            # Forward pass
            outputs = self.model(**batch_info)
            probs = F.softmax(outputs.logits, dim=-1)
            probs = probs.detach().cpu().numpy()
            predictions = np.argmax(probs, axis=-1) 

            # Append
            all_eval['gt'].append(label_ids)
            all_eval['pred'].append(predictions)
        all_eval = {k: np.array(v).flatten() for k, v in all_eval.items()}
        results_df = pd.DataFrame(all_eval)
        results_df.to_csv(os.path.join(save_dir, "results_test.csv"))


    def encode_batch(self, sentences: List[str]):
        encode_info = self.tokenizer.batch_encode_plus(
            sentences,
            padding=True,
            truncation=True,
            add_special_tokens= self.config['training']['tokenizer']['add_special_tokens'],
            max_length= self.config['training']['tokenizer']['max_length'],
            # pad_to_max_length= self.config['training']['tokenizer']['pad_to_max_length'], 
            return_tensors=self.config['training']['tokenizer']['return_tensors']
        )
        return encode_info
    

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)