import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .data import get_data_loaders
from .tokenizer_wrapper import TokenizerWrapper
from tqdm import tqdm

def evaluate_hf_model(model_name, val_loader, device, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=num_labels
    ).to(device)
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Evaluating {model_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    avg_loss = total_loss / len(val_loader)
    return avg_loss, accuracy

def compare_models(dataset_name="imdb", max_val_samples=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = TokenizerWrapper(max_length=128)
    _, val_loader, num_labels = get_data_loaders(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=32,
        max_val_samples=max_val_samples,
    )
    
    results = {}
    
    hf_models = [
        "distilbert-base-uncased",
        "bert-base-uncased",
    ]
    
    for model_name in hf_models:
        print(f"\nEvaluating {model_name}...")
        loss, accuracy = evaluate_hf_model(model_name, val_loader, device, num_labels)
        results[model_name] = {"loss": loss, "accuracy": accuracy}
        print(f"{model_name}: Loss={loss:.4f}, Accuracy={accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    results = compare_models(dataset_name="imdb", max_val_samples=1000)
    print("\n=== Comparison Results ===")
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, Loss={metrics['loss']:.4f}")
