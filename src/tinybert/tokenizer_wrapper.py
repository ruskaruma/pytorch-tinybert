from typing import List, Dict
from transformers import AutoTokenizer

class TokenizerWrapper:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_length = max_length

    def encode_batch(self, texts: List[str]) -> Dict[str, List]:
        enc = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
