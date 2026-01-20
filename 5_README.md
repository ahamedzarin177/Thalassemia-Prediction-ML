## Interpreting Early Thalassemia Risk Prediction Using CAON-Optimized Explainable Machine Learning Paradigms
This repository presents an explainable machine learning framework for early thalassemia risk prediction using clinical data. The project implements a CAON (PSO‑based) optimized XGBoost model with stratified cross‑validation to achieve reliable and reproducible performance for multi‑class disease prediction.

## Key Features
•End‑to‑end machine learning pipeline
•CAON (PSO‑based) hyperparameter optimization
•XGBoost classifier for multi‑class prediction
•Stratified 10‑fold cross‑validation
•Evaluation using Accuracy, Precision, Recall, F1‑Score, Specificity, and Cohen’s Kappa
•Clean, modular, and reproducible code structure

## Project Structure
thalassemia-prediction-ml/
│
├── data/
│   └── thalassemia.csv
│
├── notebooks/
│   └── thalassemia_prediction.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── train.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
## Workflow Overview
# Data Preprocessing
•Load dataset
•Separate features and target
•Apply global label encoding
# Modeling & Optimization
•XGBoost classifier
•CAON (Particle Swarm Optimization) for hyperparameter tuning
# Training & Evaluation
•Stratified 10‑fold cross‑validation
•Fold‑specific label handling
•Comprehensive performance metrics

## How to Run
pip install -r requirements.txt
python src/train.py

## Input
• CSV dataset containing patient clinical features
• Target column: Diagnosis (thalassemia class labels)

## Output
• Trained CAON‑optimized XGBoost model (.pkl file)
• Cross‑validated performance metrics (Accuracy, Precision, Recall, F1, Specificity, Kappa)

## The best optimized model will be saved as:
best_caon_xgb_model.pkl

## Evaluation Metrics
•Accuracy
•Precision (Macro)
•Recall (Macro)
•F1‑Score (Macro)
•Specificity
•Cohen’s Kappa Score

## Technologies Used
•Python
•NumPy, Pandas
•Scikit learn
•XGBoost
•PySwarm (PSO optimization)

## License
This project is licensed under the MIT License, allowing reuse and modification with attribution.
