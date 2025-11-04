# Architecture

## Model Overview

```mermaid
graph TD
    A[Input IDs] --> B[Embeddings]
    B --> C[Transformer Layers]
    C --> D[Pooler]
    D --> E[Classifier]
    E --> F[Logits]
```

## Embeddings Layer

```mermaid
graph LR
    A[Input IDs] --> B[Word Embeddings]
    A --> C[Position Embeddings]
    B --> D[Add]
    C --> D
    D --> E[LayerNorm]
    E --> F[Dropout]
    F --> G[Embeddings Output]
```

## Transformer Encoder Layer

```mermaid
graph TD
    A[Hidden States] --> B[Self-Attention]
    B --> C[Add & Norm]
    A --> C
    C --> D[Feed Forward]
    D --> E[Add & Norm]
    C --> E
    E --> F[Output]
```

## Self-Attention Mechanism

```mermaid
graph LR
    A[Hidden States] --> B[Q Linear]
    A --> C[K Linear]
    A --> D[V Linear]
    B --> E[Q]
    C --> F[K]
    D --> G[V]
    E --> H[MatMul & Scale]
    F --> H
    H --> I[Softmax]
    I --> J[MatMul]
    G --> J
    J --> K[Output Linear]
```

## Feed Forward Network

```mermaid
graph LR
    A[Input] --> B[Dense 1]
    B --> C[GELU]
    C --> D[Dense 2]
    D --> E[Dropout]
```

## Full Model Architecture

```mermaid
graph TD
    A[Input IDs<br/>Batch x SeqLen] --> B[Embeddings<br/>Word + Position]
    B --> C[Layer 1<br/>Attention + FFN]
    C --> D[Layer 2<br/>Attention + FFN]
    D --> E[Layer 3<br/>Attention + FFN]
    E --> F[Pooler<br/>CLS Token]
    F --> G[Dropout]
    G --> H[Classifier<br/>Linear]
    H --> I[Logits<br/>Batch x NumLabels]
```

## Classification Head

```mermaid
graph LR
    A[Pooler Output<br/>Batch x Hidden] --> B[Dropout]
    B --> C[Linear<br/>Hidden x NumLabels]
    C --> D[Logits<br/>Batch x NumLabels]
```
