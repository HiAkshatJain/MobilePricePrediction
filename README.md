# Mobile Price Prediction

Welcome to the Mobile Price Prediction project! This repository contains code for predicting mobile phone prices using various machine learning models. It includes data preprocessing, model training, and evaluation for both regression and classification tasks.

## Table of Contents

- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Data Preprocessing](#data-preprocessing)
- [Models](#models)
  - [Regression Models](#regression-models)
  - [Classification Models](#classification-models)
- [Results](#results)

## Dataset

The dataset used in this project is `Mobilephone_data_final.csv`. It contains information about mobile phones, including:

- `Brand`
- `Model`
- `Operating system`
- `Price`
- Additional features relevant to mobile phones

## Data Preprocessing

1. **Loading the Dataset**

   - The dataset is loaded into a pandas DataFrame.

2. **Data Cleaning**

   - Removed the `Sr. No.` column.
   - Selected relevant features and target variables.

3. **Feature Encoding**
   - Categorical features were encoded using Label Encoding and One-Hot Encoding.

```python
for column in ['Brand', 'Model', 'Operating system']:
    le = LabelEncoder()  # Re-initialize LabelEncoder for each column
    x[column] = le.fit_transform(x[column])  # Encode categorical column and replace in DataFrame

ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [4, 13, 14, 15, 17, 18])],  # Apply OneHotEncoder to specified columns
    remainder='passthrough'  # Leave other columns unchanged
)
```

4. **Feature Scaling**
   - Features were scaled using StandardScaler.

```python
sc = StandardScaler()  # Initialize the StandardScaler

X_train = sc.fit_transform(X_train)  # Fit to the training data and transform it
X_test = sc.transform(X_test)  # Transform the test data using the same scaler
```

## Models

### Regression Models

#### k-Nearest Neighbors (kNN) Regressor

- **Training**: Trained the kNN Regressor with `n_neighbors=5`.
- **Evaluation Metrics**:
  - RMSE: 7793.4653
  - MAE: 4079.4809
  - R-squared (R²): 0.7077

#### Linear Regression

- **Training**: Trained the model
- **Evaluation Metrics**:
  - RMSE: 7653.8744
  - MAE: 4855.3223
  - R-squared (R²): 0.7181

### Classification Models

#### Logistic Regression

- **Training**: Trained Logistic Regression for price category prediction.
- **Evaluation Metrics**:
  - Accuracy: 0.7684

```bash
Classification Report:
              precision    recall  f1-score   support

       0-10k       0.86      0.93      0.90       189
     10k-20k       0.48      0.48      0.48        50
     20k-30k       0.29      0.13      0.18        15
     30k-40k       1.00      0.11      0.20         9
        40k+       0.60      0.67      0.63         9

    accuracy                           0.77       272
   macro avg       0.65      0.46      0.48       272
weighted avg       0.76      0.77      0.75       272
```

#### k-Nearest Neighbors (kNN) Classifier

- **Training**: Trained kNN Classifier with `n_neighbors=5`.
- **Evaluation Metrics**:
  - Accuracy: 0.8088

```bash
Classification Report:
              precision    recall  f1-score   support

       0-10k       0.88      0.95      0.91       203
     10k-20k       0.45      0.51      0.48        37
     20k-30k       0.83      0.31      0.45        16
     30k-40k       0.00      0.00      0.00         8
        40k+       0.80      0.50      0.62         8

    accuracy                           0.81       272
   macro avg       0.59      0.45      0.49       272
weighted avg       0.79      0.81      0.79       272
```

## Results

Based on the evaluation metrics, the k-Nearest Neighbors model(classification) and linear(regression)achieved the highest accuracy among the models tested. Further tuning of hyperparameters could potentially enhance model performance.
