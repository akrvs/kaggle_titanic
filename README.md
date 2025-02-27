# Titanic Survival Prediction

This repository contains a machine learning project that predicts passenger survival on the Titanic using the famous Kaggle Titanic dataset. The project implements both traditional machine learning (scikit-learn) and deep learning (Keras) approaches.

## Project Structure

- `data_preprocessing.py`: Contains utilities for data cleaning and preprocessing
- `sklearn_implementation.py`: Implementation using Logistic Regression from scikit-learn
- `sklearn_test.py`: Script to generate predictions using the scikit-learn model
- `keras_implementation.py`: Implementation using a neural network built with Keras
- `keras_test.py`: Script to generate predictions using the Keras model

## Preprocessing Steps

The preprocessing pipeline handles several key tasks:
- Handling missing values in 'Age' based on 'Sex' and 'Pclass' medians
- Dropping unnecessary columns ('Cabin', 'Ticket', 'Name', 'PassengerId')
- One-hot encoding categorical features ('Sex', 'Embarked', 'Pclass')
- Optional data visualization capabilities

## Models

### Logistic Regression (sklearn)
- Simple yet effective model for binary classification
- Achieves competitive accuracy on the Titanic dataset
- Low computational requirements

### Neural Network (Keras)
- Multi-layer neural network with ReLU activation
- Architecture: Input(9) → Dense(64) → Dense(32) → Dense(16) → Dense(1/sigmoid)
- Uses early stopping to prevent overfitting

## Usage

### Prerequisites
- Python 3.x
- Required packages: pandas, numpy, scikit-learn, tensorflow, keras, matplotlib, seaborn, joblib

### Training Models

```python
# Train the scikit-learn model
python sklearn_implementation.py

# Train the Keras model
python keras_implementation.py
```

### Generating Predictions

```python
# Generate predictions with scikit-learn model
python sklearn_test.py

# Generate predictions with Keras model
python keras_test.py
```

### Data Files
The scripts are configured to look for the training and test data at specific paths:
- Training data: `/Users/akrvs/Projects/train.csv`
- Test data: `/Users/akrvs/Projects/test.csv`

You may need to modify these paths to match your local environment.

## Output Files
- `logistic_regression_model.pkl`: Saved scikit-learn model
- `keras_model.pkl`: Saved Keras model
- `submission.csv`: Predictions from the scikit-learn model
- `submission_2.csv`: Predictions from the Keras model

## Future Improvements
- Feature engineering to create more predictive variables
- Hyperparameter tuning for both models
- Ensemble methods to combine predictions
- Cross-validation for more robust evaluation
