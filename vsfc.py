import torch
from utils import load_json
from torch.utils.data import Dataset, DataLoader
class VSFCDataset(Dataset):
    def __init__(self, config, split):
        self.data = load_json(config["data"][split])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

def collate_fn(batch):
    sentences = [item["Sentence"] for item in batch]
    sentiments = [item["Sentiment"] for item in batch]
    encoded_sentiments = [item["Encoded_sentiment"] for item in batch]
    
    return sentences, sentiments, encoded_sentiments