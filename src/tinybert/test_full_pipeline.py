import torch
from pathlib import Path
import yaml
from tinybert.model import TinyBertForSequenceClassification
from tinybert.data import get_data_loaders
from tinybert.tokenizer_wrapper import TokenizerWrapper
from tinybert.train import train_epoch, evaluate
from torch.optim import AdamW

def test_full_pipeline():
    config_path = "configs/default.yaml"
    if not Path(config_path).exists():
        print("Config file not found, skipping full pipeline test")
        return
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    
    tokenizer = TokenizerWrapper(max_length=model_cfg["max_position_embeddings"])
    
    train_loader, val_loader, num_labels = get_data_loaders(
        dataset_name="imdb",
        tokenizer=tokenizer,
        batch_size=2,
        max_train_samples=20,
        max_val_samples=10,
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
    
    optimizer = AdamW(model.parameters(), lr=float(training_cfg["lr"]))
    
    train_loss = train_epoch(model, train_loader, optimizer, device)
    assert train_loss > 0
    
    val_loss, val_accuracy = evaluate(model, val_loader, device)
    assert val_loss > 0
    assert 0 <= val_accuracy <= 1
    
    print(f"Full pipeline test passed")
    print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    test_full_pipeline()
