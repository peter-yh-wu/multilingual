# Automatically Identifying Language Family from Acoustic Examples in Low Resource Scenarios

Code for respective [paper](https://arxiv.org/pdf/2012.00876).

## Background

Existing multilingual speech NLP works focus on a relatively small subset of languages, and thus current linguistic understanding of languages predominantly stems from classical approaches. In this work, we propose a method to analyze language similarity using deep learning. Namely, we train a model on the Wilderness dataset and investigate how its latent space compares with classical language family findings. Our approach provides a new direction for cross-lingual data augmentation in any speech-based NLP task.

## Quick Start

 - Download language embeddings from [here](https://drive.google.com/file/d/190kaLfQtYDEzaScb2_P2nsTkUEcxE5zf/view?usp=sharing).

 - Download Ethnologue language family trees from [here](https://drive.google.com/file/d/1wFXfhhDc2Fwk8oyhdq5VQF3W_f8XwG8P/view?usp=sharing).

 - Move `ethnologue_forest.json` to the `metadata` folder.

 - `pip3 install -r requirements.txt`

 - `cd src`

 - `python3 evaluate_embs.py -p $EMB_PATH`, where `$EMB_PATH` is the path to `embs.npy`

 - Plots and metrics will be generated in the `outputs` directory.

## Notes

 - Run `python3 build_ethnologue_tree.py` to generate `ethnologue_forest.json`.

 - The Wilderness dataset can be downloaded [here](https://github.com/festvox/datasets-CMU_Wilderness.git). Instructions to train the language classifier are described in our [paper](http://www.cs.cmu.edu/~peterw1/website_files/multilingual.pdf).

 - MCD for zero-shot TTS experiments is calculated using [Festvox](http://festvox.org/) CLUSTERGEN. 
