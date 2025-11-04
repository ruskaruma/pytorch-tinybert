import torch
from tinybert.model import TinyBertModel, TinyBertForSequenceClassification

def test_tinybert_model():
    batch_size = 2
    seq_len = 10
    vocab_size = 30522
    hidden_size = 256
    num_layers = 3
    num_heads = 4
    intermediate_size = 1024
    max_pos = 128
    
    model = TinyBertModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_pos,
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    outputs = model(input_ids, attention_mask)
    assert "last_hidden_state" in outputs
    assert "pooler_output" in outputs
    assert outputs["last_hidden_state"].shape == (batch_size, seq_len, hidden_size)
    assert outputs["pooler_output"].shape == (batch_size, hidden_size)
    
    print("TinyBERT model test passed")

def test_classification_model():
    batch_size = 2
    seq_len = 10
    vocab_size = 30522
    hidden_size = 256
    num_layers = 3
    num_heads = 4
    intermediate_size = 1024
    max_pos = 128
    num_labels = 2
    
    model = TinyBertForSequenceClassification(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_pos,
        num_labels=num_labels,
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, num_labels, (batch_size,))
    
    outputs = model(input_ids, attention_mask, labels)
    assert "loss" in outputs
    assert "logits" in outputs
    assert outputs["logits"].shape == (batch_size, num_labels)
    assert outputs["loss"] is not None
    
    outputs_no_labels = model(input_ids, attention_mask)
    assert outputs_no_labels["loss"] is None
    assert outputs_no_labels["logits"].shape == (batch_size, num_labels)
    
    print("Classification model test passed")

if __name__ == "__main__":
    test_tinybert_model()
    test_classification_model()
