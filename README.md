# Neural-Image-Captioning-with-Attention-and-Transformer-Models

# Neural Image Captioning with Attention & Transformer Models

An end-to-end deep learning project exploring multiple architectures for generating natural language descriptions from images, including attention-based RNNs and transformer-based models.

---

## Overview

This project investigates different approaches to the image captioning problem by implementing and comparing multiple neural architectures.

Rather than relying on a single model, this work focuses on understanding the trade-offs between:
- CNN + RNN (GRU/LSTM) with attention
- Object-aware captioning using detection features
- Transformer-based encoder-decoder models

The goal was to analyze how different modeling choices impact caption quality, interpretability, and performance.

---

## Problem Statement

Given an input image, generate a coherent and contextually accurate natural language description.

This requires combining:
- Visual feature extraction (computer vision)
- Sequential language generation (NLP)
- Context modeling via attention mechanisms

Image captioning is a classic multimodal problem that sits at the intersection of vision and language. :contentReference[oaicite:0]{index=0}

---

## Approach

The project is structured as a series of experiments, each implemented in separate notebooks.

### 1. Baseline Model
- CNN encoder for feature extraction
- RNN decoder (GRU/LSTM)
- Greedy decoding for caption generation

---

### 2. Attention-Based Model
- Implemented **Bahdanau attention** to allow the model to focus on relevant image regions
- Improved contextual understanding during word generation
- Attention helps models weigh important features dynamically :contentReference[oaicite:1]{index=1}

---

### 3. Transformer-Based Model
- Implemented encoder-decoder transformer architecture
- Replaced recurrence with **self-attention mechanisms**
- Enabled better parallelization and long-range dependency modeling :contentReference[oaicite:2]{index=2}

---

### 4. Experimental Variations
- Explored different CNN backbones for feature extraction
- Compared decoding strategies (greedy vs beam search)
- Evaluated performance across architectures

---

## Dataset

- Flickr8k dataset used for training and evaluation
- Images paired with multiple human-written captions

---

## Evaluation

- BLEU scores used to measure caption quality
- Qualitative analysis of generated captions

---

## Tech Stack

- Python  
- PyTorch  
- Jupyter Notebooks  
- NumPy / Pandas  

---

## Key Insights

- Attention mechanisms significantly improve caption relevance
- Transformer models outperform RNN-based models on longer sequences
- Beam search improves fluency compared to greedy decoding
- Model performance is highly sensitive to feature extraction quality

---

## Results

- Attention-based models produced more context-aware captions
- Transformer-based models generated more coherent and fluent sentences
- Observed improvements in BLEU scores across advanced architectures

---

## Notes

- Dataset not included due to size constraints
- Google Drive used for dataset storage and intermediate features

---

## Future Work

- Train on larger datasets (Flickr30k, MS-COCO)
- Integrate object detection (e.g., YOLO) for improved semantics
- Explore multimodal pretraining (CLIP-style models)

---

## Acknowledgements

- Flickr8k dataset  
- Research on attention mechanisms and transformers  
