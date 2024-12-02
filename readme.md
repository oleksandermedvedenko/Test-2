# Machine Learning Regression Project

This project implements a machine learning solution for predicting a target variable using 53 anonymized features. The model is optimized for RMSE (Root Mean Square Error).

## Project Structure
- exploratory_analysis.ipynb: Jupyter notebook for data analysis
- train.py: Script for model training
- predict.py: Script for making predictions
- requirements.txt: Project dependencies
- model.joblib: Trained model (generated after training)
- scaler.joblib: Feature scaler (generated after training)

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Place your training data as 'train.csv' in the project root
4. Place your test data as 'hidden_test.csv' in the project root

## Usage
1. Run exploratory analysis: `jupyter notebook exploratory_analysis.ipynb`
2. Train the model: `python train.py`
3. Generate predictions: `python predict.py`

The predictions will be saved in 'predictions.csv'.