# Representation Profiling under Augmentations

## Overview
This project provides a modular framework to analyze how internal representations of deep neural networks change under input transformations (augmentations).

It enables:

- Identification of **sensitive layers**
- Measurement of **representation stability**
- Evaluation of **embedding robustness**

The framework is designed to be **modular, reproducible, and aligned with research practices**.


## Key Idea
The framework quantifies how internal representations shift when inputs are transformed, enabling:

- detection of augmentation-specific failure modes  
- understanding of layer-wise robustness  
- analysis of feature stability  


## Features

- Per-augmentation analysis (no mixing of effects)  
- Layer-wise feature extraction via hooks  
- Multiple evaluation metrics:
  - CKA (Centered Kernel Alignment)
  - Cosine Similarity
  - Normalized L2 Distance
  - Gram Matrix Similarity (structure-aware)
- Composite sensitivity scoring  
- Embedding-level robustness metric  
- Config-driven pipeline  
- Deterministic and reproducible execution  


## Structure Overview

- **profiler/** — Core module handling representation extraction, metric computation, and reporting  
- **augment.py** — Applies input perturbations for sensitivity analysis  
- **hooks.py** — Registers forward hooks for feature extraction  
- **profile_model.py** — Main script to run profiling pipeline  
## Installation

```bash
pip install torch torchvision kornia numpy
```

## Usage

Run the profiling pipeline:

```bash
python profile_model.py
```

### This will:

- apply augmentations
- extract features
- compute metrics
- print results
- save `report.json`
## API Usage

```python
from profiler.engine import model_profile_under_augmentation

results = model_profile_under_augmentation(model, config)
```

## Configuration

```python
config = {
    "layers": ["layer1", "layer2", "layer3", "layer4"],
    "augmentations": [
        {"name": "rotation", "params": {"degrees": 15}},
        {"name": "blur", "params": {}},
        {"name": "brightness", "params": {}},
    ],
    "mode": "individual",
    "top_k": 2,
    "processing": "flatten",
    "input": torch.rand(1, 3, 224, 224)
}
```
## Metrics

### Sensitivity

Measures how much a layer changes under augmentation:

- **CKA** → representation similarity  
- **L2_normalized** → magnitude of change  
- **Composite Score** = (1 - CKA) + L2_normalized  

### Stability

- Cosine similarity  
- Gram similarity (if spatial features are used)  

### Embedding Robustness

- **R** = cosine_similarity * exp(-L2_normalized)  


### Example Output

```text
=== AUGMENTATION: RandomRotation ===

Top Sensitive Layers:
  - layer4
  - layer2

Sensitivity:
  layer4: CKA=0.3059, L2_norm=1.7468, Score=2.4409

Embedding Robustness:
  Cosine=0.4899, L2_norm=1.6093, Score=0.0980
```


## Insights Enabled

- Which augmentations disrupt representations  
- Which layers are most sensitive  
- How stable intermediate features are  
- Whether embeddings remain robust  


## Reproducibility

All experiments are reproducible via:

- fixed random seeds  
- deterministic PyTorch execution  


## Design Principles

- Modular architecture  
- Clear separation of concerns  
- Config-driven execution  
- Research-aligned evaluation  


## Future Work

- Dataset-level evaluation  
- Visualization dashboards  
- Additional metrics   


## Summary

This framework provides a structured and reproducible way to analyze representation behavior under augmentations, enabling deeper understanding of model robustness and internal dynamics.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
