# TDA for Protein Large Language Models

This repository contains the implementation for the project "Topological Data Analysis for Protein Large Language Models" which explores the topological properties of attention maps in transformer-based protein language models (pLMs) to improve protein characterization.

## Dataset

We use the Gene Ontology (GO) dataset, which contains information about protein functions across different levels of biological systems:
- Molecular Function (MF): Activity of a gene product at the molecular level
- Cellular Component (CC): Cellular localization and association with cellular compartments
- Biological Process (BP): Affiliation with larger mechanisms and pathways

Dataset splits:
- Train: 26,225 samples
- Validation: 2,904 samples
- Test: 3,350 samples

Processed dataset is available on Hugging Face: https://huggingface.co/datasets/ultracheese/tda_for_proteins

## Methods

Our approach involves several steps:
1. Extract attention maps from the ESM-2 protein language model
2. Convert attention maps into weighted graphs
3. Construct Minimum Spanning Trees (MSTs) from these graphs
4. Extract topological features (barcodes) from the MSTs
5. Vectorize the barcodes using various methods:
   - Descriptive statistics
   - Betti Curves
   - Persistence Images
   - Persistence Landscapes
6. Train classifiers (KNN, MLP, PyBoost) on these features

## Installation

```bash
git clone https://github.com/itsultracheese/tda_per_proteins.git
cd tda_per_proteins

pip install -r requirements.txt
```

## Usage

Each of the methods is available in it's own directory. All of the scripts are self-explanatory and ready-to-use, only make sure to plug-in correct file paths.

## Team Members

- Pavel Borisenko: Persistence landscapes and classification
- Dmitrii Iarchuk: Statistical features from barcodes and classification
- Roman Makarov: Persistence images and feature extraction using MobileNetV3
- Amina Miftahova: Betti curves and classification, comparison of homology approaches
- Dmitry Petrov: Dataset preprocessing, attention maps extraction, and barcodes calculation

## Supervisors

- Maria Ivanova
- Alexander Mironenko
