import click
import torch
from pathlib import Path
import yaml
from .train import train, evaluate
from .model import TinyBertForSequenceClassification
from .data import get_data_loaders
from .tokenizer_wrapper import TokenizerWrapper

@click.command()
@click.option("--config", default="configs/default.yaml", help="Path to config file")
@click.option("--dataset", default="imdb", type=click.Choice(["imdb", "ag_news"]), help="Dataset to use")
@click.option("--mode", default="train", type=click.Choice(["train", "eval", "compare"]), help="Mode: train, eval, or compare")
@click.option("--checkpoint", default=None, help="Path to model checkpoint for eval")
def main(config, dataset, mode, checkpoint):
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"Config file not found: {config}")
        return
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]
    
    tokenizer = TokenizerWrapper(max_length=model_cfg["max_position_embeddings"])
    train_loader, val_loader, num_labels = get_data_loaders(
        dataset_name=dataset,
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
    
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    
    if mode == "train":
        from .train import train_epoch
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import LinearLR
        
        optimizer = AdamW(model.parameters(), lr=float(training_cfg["lr"]), weight_decay=float(training_cfg["weight_decay"]))
        total_steps = len(train_loader) * training_cfg["epochs"]
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
        
        for epoch in range(training_cfg["epochs"]):
            click.echo(f"\nEpoch {epoch + 1}/{training_cfg['epochs']}")
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, device)
            
            click.echo(f"Train Loss: {train_loss:.4f}")
            click.echo(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            
            for _ in train_loader:
                scheduler.step()
            
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            click.echo(f"Saved checkpoint: {checkpoint_path}")
    elif mode == "compare":
        from .compare_hf import compare_models
        results = compare_models(dataset_name=dataset, max_val_samples=1000)
        click.echo("\n=== Comparison Results ===")
        for model_name, metrics in results.items():
            click.echo(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, Loss={metrics['loss']:.4f}")
        tinybert_val_loss, tinybert_val_accuracy = evaluate(model, val_loader, device)
        click.echo(f"\nTinyBERT (From Scratch): Accuracy={tinybert_val_accuracy:.4f}, Loss={tinybert_val_loss:.4f}")
    else:
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        click.echo(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

if __name__ == "__main__":
    main()
