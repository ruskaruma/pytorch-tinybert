# src/tinybert/embeddings.py
from typing import Optional
import torch
import torch.nn as nn

class TinyBertEmbeddings(nn.Module):
    """
    Learned token embeddings + learned position embeddings similar to BERT.
    Expects input_ids shape: (batch_size, seq_len)
    Returns embeddings shape: (batch_size, seq_len, hidden_size)
    """
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.layernorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self._max_pos = max_position_embeddings

    def forward(self, input_ids: torch.LongTensor, position_ids: Optional[torch.LongTensor] = None):
        """
        input_ids: (B, T)
        position_ids: Optional (B, T); if None, created automatically [0..T-1]
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids should be shape (batch_size, seq_len)")
        bsz, seq_len = input_ids.size()
        if seq_len > self._max_pos:
            raise ValueError(f"Sequence length {seq_len} exceeds max_position_embeddings {self._max_pos}")

        if position_ids is None:
            device = input_ids.device
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, seq_len)

        word_emb = self.word_embeddings(input_ids)           # (B, T, H)
        pos_emb = self.position_embeddings(position_ids)     # (B, T, H)
        embeddings = word_emb + pos_emb
        embeddings = self.layernorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
