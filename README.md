# Predictive Maintenance — Equipment Failure Classification

This folder contains a Jupyter notebook that explores predictive maintenance for industrial equipment. The notebook trains and evaluates classifiers to predict failure events using sensor and operational data.

## Notebook
- `predictive_maintenance.ipynb` — data loading, preprocessing, encoding, scaling, oversampling (SMOTE/SMOTEENN), model training (Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting) and evaluation (precision, recall, F1, confusion matrices).


## Key steps in the notebook
- Load dataset and inspect label distribution
- Drop identifiers and highly correlated features to avoid leakage
- Encode categorical `Type` feature via one-hot encoding
- Scale numeric features with `StandardScaler`
- Oversample the minority class using `SMOTEENN` (combined over- and under-sampling)
- Train and evaluate multiple classifiers and compare precision/recall/F1 for the positive (failure) class

## Limitations
- Results are illustrative; hyperparameter tuning and proper cross‑validation should be added for production use.
- Oversampling must be performed only on the training set (not on test data) as done in the notebook.

