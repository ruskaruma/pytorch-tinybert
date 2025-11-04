from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from .tokenizer_wrapper import TokenizerWrapper

class TextClassificationDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer: TokenizerWrapper):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer.encode_batch([text])
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": torch.tensor(label, dtype=torch.long),
        }

def collate_fn(batch):
    max_len = max(item["input_ids"].size(0) for item in batch)
    batch_size = len(batch)
    
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels_list = []
    
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].size(0)
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels_list.append(item["labels"])
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.stack(labels_list),
    }

def load_imdb_dataset(tokenizer: TokenizerWrapper, split: str = "train", max_samples: int = None):
    dataset = load_dataset("imdb", split=split)
    texts = dataset["text"]
    labels = dataset["label"]
    
    if max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
    
    return TextClassificationDataset(texts, labels, tokenizer)

def load_agnews_dataset(tokenizer: TokenizerWrapper, split: str = "train", max_samples: int = None):
    dataset = load_dataset("ag_news", split=split)
    texts = [item["text"] for item in dataset]
    labels = dataset["label"]
    
    if max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
    
    return TextClassificationDataset(texts, labels, tokenizer)

def get_data_loaders(dataset_name: str, tokenizer: TokenizerWrapper, batch_size: int, max_train_samples: int = None, max_val_samples: int = None):
    if dataset_name == "imdb":
        train_dataset = load_imdb_dataset(tokenizer, "train", max_train_samples)
        val_dataset = load_imdb_dataset(tokenizer, "test", max_val_samples)
        num_labels = 2
    elif dataset_name == "ag_news":
        train_dataset = load_agnews_dataset(tokenizer, "train", max_train_samples)
        val_dataset = load_agnews_dataset(tokenizer, "test", max_val_samples)
        num_labels = 4
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, num_labels
