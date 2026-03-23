# Representation Profiling Framework

A modular framework to analyze how internal representations of deep neural networks change under input transformations.

This framework provides a generic, extensible interface for studying representation behavior across layers, enabling robustness analysis, feature stability evaluation, and transformation sensitivity studies.


## Overview

Modern deep learning models often behave unpredictably under input changes such as augmentations, corruptions, or domain shifts. This framework enables systematic analysis of how internal representations evolve under such transformations.

It is designed to be:

- modular  
- reproducible  
- extensible  
- aligned with deep learning framework design principles  


## Core API

The framework provides a generic interface for comparing representations under arbitrary input transformations:

```python
from profiler.api import model_profiling_under_input_changes

profiler = model_profiling_under_input_changes(
    model,
    input_a,
    input_b,
    config=config
)

df = profiler.df
```
For lower-level control, the core profiler can be used directly:

```python
from profiler.model_profiler import ModelProfiler

with ModelProfiler(
    model,
    layers=config.get("layers"),
    processing=config.get("processing", "flatten")
) as p:
    p(x=input_a, group="group_a")
    p(x=input_b, group="group_b")

df = p.df
```

This abstraction enables flexible analysis of representation changes without being tied to any specific transformation.

## Augmentation Profiling (Wrapper)

Augmentation-based analysis is implemented as a thin wrapper over the core API:

```python
from profiler.api import model_profile_under_augmentation

results = model_profile_under_augmentation(model, config)
```

This keeps the core profiler generic while supporting common use cases such as robustness evaluation under augmentations.

## Features

- Generic input-to-input representation profiling (`input_a → input_b`)
- Context-manager based execution (PyTorch-style)
- Group-based comparison of representations
- Layer-wise feature extraction via forward hooks
- Modular and extensible metric system
- Config-driven pipeline (supports dict and JSON)
- Structured outputs with pandas integration
- Augmentation support implemented as a wrapper (not a core dependency)

## Project Structure

- `profiler/` — Core profiling logic (API, model profiler, metrics, reporting)  
- `augment.py` — Input transformation utilities  
- `hooks.py` — Feature extraction via forward hooks  
- `profile_model.py` — Example script for running profiling pipeline  

## Installation

```bash
pip install torch torchvision kornia numpy
```

## Usage

### Core Profiling

```python
from profiler.api import model_profiling_under_input_changes

profiler = model_profiling_under_input_changes(
    model,
    input_a,
    input_b,
    config=config
)

df = profiler.df
```

### Augmentation Profiling

```bash
python profile_model.py
```

## Configuration

The framework supports both Python dictionaries and JSON configuration files.

Example:

```python
config = {
    "layers": ["layer1", "layer2", "layer3", "layer4"],
    "augmentations": [
        {"name": "rotation", "params": {"degrees": 15}},
        {"name": "blur", "params": {}},
        {"name": "brightness", "params": {}}
    ],
    "mode": "individual",
    "processing": "flatten",
    "input": torch.rand(1, 3, 224, 224)
}
```

You can also pass a JSON file path:

```python
config = "config.json"
```

## Metrics

The framework supports modular, composable metrics such as:

- Cosine Similarity  
- CKA (Centered Kernel Alignment)  
- Normalized L2 Distance  
- Gram Matrix Similarity  

Higher-level analysis metrics (e.g., sensitivity, stability) can be built on top of these primitives.

## Output

Results are accessible as a pandas DataFrame:

```python
df = profiler.df
```

Optionally, results can be saved to a file:

```python
model_profiling_under_input_changes(..., output="report.json")
```

The output contains layer-wise comparisons across groups and metrics.

## Design Philosophy

- Separation of concerns: core profiler is independent of input transformations  
- Composable metrics: atomic metrics enable flexible analysis  
- Reproducibility: config-driven execution  
- Framework alignment: API design inspired by PyTorch profiler  
- Minimalism: avoids overengineering while remaining extensible  

## Future Work

- Dataset-level evaluation  
- Visualization dashboards  
- Additional metrics and analysis tools  
- Integration with training pipelines  

## License

This project is licensed under the MIT License.
