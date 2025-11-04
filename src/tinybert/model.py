import torch
import torch.nn as nn
from .embeddings import TinyBertEmbeddings
from .transformer import TransformerEncoderLayer

class TinyBertModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embeddings = TinyBertEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_hidden_layers)
        ])
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = self.activation(pooled_output)
        
        return {
            "last_hidden_state": hidden_states,
            "pooler_output": pooled_output,
        }

class TinyBertForSequenceClassification(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bert = TinyBertModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs["pooler_output"]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
        }
