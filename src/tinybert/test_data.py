import torch
from tinybert.data import TextClassificationDataset, load_imdb_dataset, load_agnews_dataset
from tinybert.tokenizer_wrapper import TokenizerWrapper

def test_text_classification_dataset():
    tokenizer = TokenizerWrapper(max_length=128)
    texts = ["This is a test sentence.", "Another example text here."]
    labels = [0, 1]
    
    dataset = TextClassificationDataset(texts, labels, tokenizer)
    assert len(dataset) == 2
    
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["input_ids"].shape[0] > 0
    assert sample["labels"].item() == 0
    
    print("TextClassificationDataset test passed")

def test_imdb_dataset():
    tokenizer = TokenizerWrapper(max_length=128)
    dataset = load_imdb_dataset(tokenizer, "train", max_samples=10)
    
    assert len(dataset) == 10
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["labels"].item() in [0, 1]
    
    print("IMDB dataset test passed")

def test_agnews_dataset():
    tokenizer = TokenizerWrapper(max_length=128)
    dataset = load_agnews_dataset(tokenizer, "train", max_samples=10)
    
    assert len(dataset) == 10
    sample = dataset[0]
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert sample["labels"].item() in [0, 1, 2, 3]
    
    print("AG News dataset test passed")

if __name__ == "__main__":
    test_text_classification_dataset()
    test_imdb_dataset()
    test_agnews_dataset()
