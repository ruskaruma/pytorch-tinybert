import torch
import torch.nn as nn
from .attention import SelfAttention

class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, hidden_states):
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.attention = SelfAttention(hidden_size, num_heads, dropout)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.layernorm_1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.layernorm_2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, hidden_states, attention_mask=None):
        attn_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.layernorm_1(hidden_states + self.dropout_1(attn_output))
        
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.layernorm_2(hidden_states + self.dropout_2(ffn_output))
        
        return hidden_states
