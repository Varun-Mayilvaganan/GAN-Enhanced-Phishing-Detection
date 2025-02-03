# GAN-Enhanced Phishing Detection: Improving Classifier Robustness with Synthetic Malicious Data 

## Introduction

Phishing attacks have become one of the most prevalent cybersecurity threats, tricking users into divulging sensitive information through fraudulent websites. Traditional phishing detection systems often struggle with evolving attack techniques, leading to a growing need for more robust solutions. This project introduces a machine learning-based phishing detection system that leverages Gradient Boosting Machines (GBM), XGBoost, LightGBM, and CatBoost to classify websites as legitimate or phishing. Additionally, Generative Adversarial Networks (GANs) are employed to generate synthetic malicious data, improving classifier resilience against adversarial attacks and previously unseen phishing patterns.

## Overview

A robust phishing detection framework that leverages machine learning techniques, including adversarial training and multi-modal feature extraction. By analyzing diverse data sources such as URLs, HTML content, metadata, and user behavior, This project uses GAN generated samples to avoid adversarial attacks.

## Project Structure
```
|_ README.md                     # Project overview, setup, and usage instructions
|_ requirements.txt              # List of dependencies
|_ data/
   |_ raw/                       # Raw dataset files
   |_ processed/                  # Processed dataset files
|_ notebooks/                    
   |_ Phish_Legit_FeatureEngineering.ipynb  # Feature engineering notebook
   |_ Legit_and_Phish_Analysis.ipynb         # Data analysis notebook
   |_ Model_training.ipynb        # models and its training
|_ scripts/
   |_ feature_engineering.py      # Script for feature engineering
   |_ gan_samples_generate.py     # generates GAN samples
|_ results/
   |_ output.csv              # Comparison of all models 
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Varun-Mayilvaganan/GAN-Enhanced-Phishing-Detection
   cd GAN-Enhanced-Phishing-Detection
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### 1. Data Processing
Place your raw dataset in `data/raw/` and run the feature engineering script or else you can use it in google colaboratory:
```bash
python scripts/feature_engineering.py
```
This will generate a processed dataset in `data/processed/`.

### 2. Detailed exploration of data
```
|_notebooks
  |_Legit_and_Phish_Analysis.ipynb
```

### 3. ML models and training
The below notebook has the traditional machine learning models and its training.
```
|_notebooks
  |_Model_training.ipynb
```

## Dependencies
- Python 3
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- XGBoost
- LightGBM
- CatBoost
- PyTorch (for GAN implementation)

## Results
The output logs and accuracy of classification metrics will be stored in `results/output.csv`.

## Future Enhancements
- Deploy as a web service using Flask or FastAPI
- Improve feature extraction with deep learning techniques
- Integrate real-time phishing URL detection
- Data seems to be overfit so, trying to minimize the loss


Feel free to contribute or report any issues!


