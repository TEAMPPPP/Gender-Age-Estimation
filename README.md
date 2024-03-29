# Gender-Age Estimation

This project aims to estimate gender and age from images using deep learning techniques.

## Installation

Follow the steps below to set up the environment:

Create a new Conda environment:
   ```bash
   conda create -n GAE python==3.8
   conda activate GAE
   cd AGE_gender_estimation
   ```
Install PyTorch and related dependencies with CUDA 11.7 support:
   ```bash
   conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```
Install other required libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

## UTKface Datasets (Classification: age / gender)
- download link : [https://drive.google.com/drive/folders/1IJC0Z1XyOr7E0StbeYWvPn4uTHxHIXwf?usp=drive_link](https://www.kaggle.com/datasets/jangedoo/utkface-new?rvi=1)

## Weights Downloads
Create a pretrained folder and place the pths in it.
- download link : https://drive.google.com/drive/folders/1YeYffSZx0m8m02q4KtfNXnpCg4iamqry

## Lets Start!!!
To train, run train.py (However, please configure the settings according to the user's interface before proceeding.)
   ```bash
   python train.py
   ```

To perform real-time demo processing, run Observer.py
   ```bash
   python Observer.py
   ```

## TO-DO List
- ResNet50 test training [O]
- SwinTransformer v2 training [X]
- Dockerfile generation [X]
- FastAPI [X]