# Predict Customer Engagement in E-commerce

ML project from my MSc at Western University. The goal was to predict whether an online shopper would make a purchase based on their session data.

## The Problem

Only 17% of online shoppers visit e-commerce sites with the intention of buying something. We used session data from 12,330 unique visitors to build models that predict purchase intent — which can help businesses target the right customers at the right time.

## Dataset

- 12,330 sessions, each from a unique user over a 1-year period
- 18 features including page views, session duration, traffic type, browser, region, and more
- Heavy class imbalance: 7,302 non-buyers vs 1,445 buyers (83.5% vs 16.5%)

## What We Did

- Cleaned and preprocessed the data (median imputation, one-hot encoding, normalization)
- Applied SMOTE upsampling on the training set to handle class imbalance
- Trained and tuned 8 models: SVM, KNN, Random Forest, MLP, AdaBoost, Bagging, Gradient Boosting, and Voting Classifier
- Used 10-fold cross-validation for hyperparameter tuning
- Evaluated models on accuracy, precision, recall, F1, and AUROC

## Results (Upsampled Training Data)

| Model | Accuracy | AUROC |
|---|---|---|
| Gradient Boosting | 90% | 0.83 |
| Voting Classifier | 90% | 0.84 |
| Bagging | 89% | 0.85 |
| Random Forest | 88% | 0.85 |
| AdaBoost | 89% | 0.82 |

Bagging with a tuned Random Forest as its base estimator performed best overall. Upsampling improved recall and AUROC for all models except MLP.

## Tech used

- Python, Scikit-learn, Pandas, NumPy, Matplotlib
- SMOTE (imbalanced-learn)
- Jupyter Notebook
