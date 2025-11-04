import torch
from tinybert.transformer import TransformerEncoderLayer

def test_transformer_layer():
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    num_heads = 4
    intermediate_size = 1024
    
    layer = TransformerEncoderLayer(hidden_size, num_heads, intermediate_size)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = layer(hidden_states, attention_mask)
    assert output.shape == (batch_size, seq_len, hidden_size), f"Expected shape ({batch_size}, {seq_len}, {hidden_size}), got {output.shape}"
    
    print("Transformer encoder layer test passed")

if __name__ == "__main__":
    test_transformer_layer()
