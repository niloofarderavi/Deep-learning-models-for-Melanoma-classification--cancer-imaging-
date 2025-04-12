This repository contains a comparative study between Swin Transformer and ResNet-50 models for melanoma skin cancer classification using a dataset of 10,000 images.

Models
Swin Transformer: Swin-B (microsoft/swin-base-patch4-window7-224-in22k) pre-trained on ImageNet-22k

ResNet: ResNet-50 pre-trained on ImageNet-1k with ResNet50_Weights.IMAGENET1K_V2 weights

Dataset
The Melanoma Skin Cancer Dataset of 10,000 Images from Kaggle is used, containing:

Benign and malignant skin lesion images

Train/test split provided by the dataset

Additional validation split created from training data (80/20 split)

Key Features
Comprehensive data augmentation pipeline

Class imbalance handling through oversampling

Enhanced loss function with class weighting

Detailed evaluation metrics (accuracy, precision, recall, F1, AUC-ROC)

Threshold optimization for classification

McNemar's test for statistical significance comparison

Training history visualization

Requirements
Python 3.7+

PyTorch

torchvision

transformers

scikit-learn

matplotlib

tqdm

kagglehub

statsmodels

Usage
Install dependencies: pip install -r requirements.txt

Run the main script: python melanoma_classification.py

The script will:

Download and preprocess the dataset

Train both Swin Transformer and ResNet-50 models

Evaluate on the test set

Generate performance metrics and visualizations

Compare model performance statistically

Results
The script outputs:

Training/validation metrics for each epoch

Test set performance with default (0.5) and optimized thresholds

Confusion matrices for both models

Statistical significance test between models

Training history plots (loss, accuracy, etc.)

Output Files
best_model_swin.pth - Best performing Swin Transformer weights

best_model_resnet.pth - Best performing ResNet-50 weights

training_history_swin.png - Swin training metrics visualization

training_history_resnet.png - ResNet training metrics visualization

Note
The dataset is automatically downloaded using the KaggleHub API. Ensure you have proper Kaggle credentials configured.
