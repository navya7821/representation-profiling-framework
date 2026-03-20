# Representation Profiling Framework

A modular framework for analyzing how deep neural network representations respond to input perturbations and augmentations.

This project focuses on feature-level behavior, going beyond simple output comparison to provide structured insights into layer sensitivity, representation stability, and model robustness.

## Overview

Modern vision models often behave unpredictably under input transformations. This framework provides tools to inspect intermediate feature representations, quantify how they change under augmentations, identify sensitive layers, and evaluate robustness using multiple similarity metrics.

## Key Features

Feature Extraction via Hooks
Layer-wise feature capture using forward hooks with a modular FeatureExtractor abstraction. Supports arbitrary PyTorch models.

Augmentation Pipeline
Built using Kornia augmentations with an easily extendable transformation pipeline.

Multi-Metric Representation Analysis
Instead of relying on a single metric, this framework combines CKA (representation similarity), normalized L2 distance (magnitude of change), cosine similarity (directional alignment), and Gram matrix similarity (structural consistency).

Layer Sensitivity Profiling
Computes sensitivity scores per layer, ranks layers based on transformation impact, and identifies the most affected regions in the network.

Representation Stability Analysis
Evaluates how stable internal representations remain using cosine and Gram-based comparisons.

Embedding Robustness Score
Measures final representation consistency by combining cosine similarity and normalized distance.

## Why This Matters

Most tools only measure whether the output changes. This framework focuses on where the change occurs, how significant it is, which layers are most sensitive, and how robust the model is overall.

## Project Structure

- `hooks.py` — Feature extraction module  
- `augment.py` — Augmentation pipeline  
- `main.py` — End-to-end analysis pipeline  

## Usage

Import the modules and run the main pipeline:

python main.py

## Current Status

This is the baseline version of the framework.

Planned extensions include dataset integration for large-scale evaluation, semantic or invariant-based validation, and visualization tools for layer-wise behavior.

## Future Direction

This framework is being extended toward robustness benchmarking, model debugging, interpretability, and deployment-time validation for detecting silent failures.

## License

MIT License
