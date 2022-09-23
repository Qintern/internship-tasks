# Task 4 - detecting soil erosion

## Content 
- data_preparation.py - preprocessing jp2 file with corresponding masks into 224x224 images in data/ dir
- dataset.py - PyTorch Dataset collecting all images for training and testing
- model.py - UNet model to train
- train.py - PyTorch training loop (with validating)
- eval.py - Evaluating trained model

## Requirements
``pip install -r requirements.txt``

## How To Run
1. Data preprocessing: `python data_preparation.py`
2. Model training: `python train.py`
3. Evaluating: `python eval.py`

