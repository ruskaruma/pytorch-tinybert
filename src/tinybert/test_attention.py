import torch
from tinybert.attention import SelfAttention

def test_self_attention():
    batch_size = 2
    seq_len = 10
    hidden_size = 256
    num_heads = 4
    
    attention = SelfAttention(hidden_size, num_heads)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = attention(hidden_states, attention_mask)
    assert output.shape == (batch_size, seq_len, hidden_size), f"Expected shape ({batch_size}, {seq_len}, {hidden_size}), got {output.shape}"
    
    attention_mask[:, 5:] = 0
    output_masked = attention(hidden_states, attention_mask)
    assert output_masked.shape == (batch_size, seq_len, hidden_size)
    
    print("Self-attention test passed")

if __name__ == "__main__":
    test_self_attention()
