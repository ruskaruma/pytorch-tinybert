import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from .model import TinyBertForSequenceClassification
from .data import get_data_loaders
from .tokenizer_wrapper import TokenizerWrapper

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

def train(config_path: str = "configs/default.yaml"):
    from pathlib import Path
    import yaml
    import random
    import numpy as np
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    seed = cfg["project"]["seed"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    
    tokenizer = TokenizerWrapper(max_length=model_cfg["max_position_embeddings"])
    
    train_loader, val_loader, num_labels = get_data_loaders(
        dataset_name="imdb",
        tokenizer=tokenizer,
        batch_size=training_cfg["batch_size"],
    )
    
    model = TinyBertForSequenceClassification(
        vocab_size=model_cfg["vocab_size"],
        hidden_size=model_cfg["hidden_size"],
        num_hidden_layers=model_cfg["num_hidden_layers"],
        num_attention_heads=model_cfg["num_attention_heads"],
        intermediate_size=model_cfg["intermediate_size"],
        max_position_embeddings=model_cfg["max_position_embeddings"],
        num_labels=num_labels,
        dropout=model_cfg["dropout"],
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=float(training_cfg["lr"]), weight_decay=float(training_cfg["weight_decay"]))
    
    total_steps = len(train_loader) * training_cfg["epochs"]
    warmup_steps = training_cfg["warmup_steps"]
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    
    for epoch in range(training_cfg["epochs"]):
        print(f"\nEpoch {epoch + 1}/{training_cfg['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        for _ in train_loader:
            scheduler.step()
    
    return model
