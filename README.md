# PyTorch-TinyBERT

From-scratch implementation of a lightweight BERT-like transformer for text classification. Features multi-head self-attention, GELU-based feed-forward layers, residual connections, and attention masking.

## Features

- Modular architecture (attention.py, transformer.py, model.py)
- From scratch implementation (no pretrained weights)
- Support for IMDB and AG News datasets
- CLI interface for training and evaluation
- Comparison with Hugging Face pretrained models

## Installation

```bash
pip install -e .
```

## Quick Start

**Training:**
```bash
python -m tinybert.cli --mode train --dataset imdb
```

**Evaluation:**
```bash
python -m tinybert.cli --mode eval --dataset imdb --checkpoint checkpoint_epoch_3.pt
```

**Compare:**
```bash
python -m tinybert.cli --mode compare --dataset imdb
```

## Model Architecture

- 3 transformer encoder layers, 256 hidden size, 4 attention heads
- 1024 intermediate size, 128 max sequence length
- ~12M parameters

See [architecture.md](architecture.md) for detailed diagrams.

## Results - IMDB Sentiment Classification

| Model | Accuracy | Parameters | Notes |
|-------|----------|------------|-------|
| TinyBERT (From Scratch) | 75.20% | ~12M | Verified, 3 epochs training |
| DistilBERT | ~90%+ | 66M | Expected (pretrained + fine-tuned) |
| BERT-base | ~90%+ | 110M | Expected (pretrained + fine-tuned) |

**Training Progress:** Epoch 1: 70.65% → Epoch 2: 72.95% → Epoch 3: 75.20% (loss: 0.56 → 0.50)

*Note: HF model comparisons require fine-tuning on IMDB. The numbers shown are expected performance based on literature. Use `--mode compare` to benchmark after fine-tuning.*

## License

MIT License