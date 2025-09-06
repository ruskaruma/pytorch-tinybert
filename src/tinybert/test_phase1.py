# src/tinybert/test_phase1.py
"""
Sanity test for Phase 1: verifies embeddings forward pass shapes and numeric outputs.
Run:
    source project-env/bin/activate
    python -m src.tinybert.test_phase1
or from project root:
    python src/tinybert/test_phase1.py
"""
import torch
from pathlib import Path
import yaml
from tinybert.embeddings import TinyBertEmbeddings
from tinybert.tokenizer_wrapper import TokenizerWrapper

def load_config(path="configs/default.yaml"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with p.open("r") as fh:
        return yaml.safe_load(fh)

def main():
    cfg = load_config()
    model_cfg = cfg["model"]
    training_cfg = cfg["training"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Tokenizer for creating input ids (fast path for sanity)
    tokenizer = TokenizerWrapper(max_length=model_cfg.get("max_position_embeddings", 128))
    samples = [
        "This is a short test sentence.",
        "Second example to verify batches and padding behavior."
    ]
    enc = tokenizer.encode_batch(samples)
    input_ids = enc["input_ids"].to(device)  # (B, T)
    attention_mask = enc["attention_mask"].to(device)

    vocab_size = model_cfg.get("vocab_size", 30522)
    hidden_size = model_cfg.get("hidden_size", 256)
    max_pos = model_cfg.get("max_position_embeddings", 128)

    embeddings = TinyBertEmbeddings(vocab_size=vocab_size, hidden_size=hidden_size, max_position_embeddings=max_pos).to(device)
    embeddings.eval()

    with torch.no_grad():
        out = embeddings(input_ids)  # (B, T, H)

    print("input_ids shape:", input_ids.shape)
    print("embeddings output shape:", out.shape)
    # print a small numeric sample for sanity
    print("embeddings[0,0,:5]:", out[0, 0, :5].cpu().numpy())

    # Quick shape assertions (will raise if wrong)
    assert out.shape[0] == input_ids.shape[0], "batch dim mismatch"
    assert out.shape[1] == input_ids.shape[1], "sequence length mismatch"
    assert out.shape[2] == hidden_size, "hidden size mismatch"

    print("Phase 1 sanity test passed.")

if __name__ == "__main__":
    main()
