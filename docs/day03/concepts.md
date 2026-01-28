# Day 3: The Complete Workflow - Train/Test Splits & Metrics

## Why We Split Data

The fundamental principle of machine learning evaluation is that we want our model to **generalize** well to new, unseen data - not just memorize the training data.

### The Problem with No Splitting

If you train and evaluate your model on the same data:
- Your model might achieve perfect accuracy on training data
- But perform terribly on new data (customers, users, real-world scenarios)
- This is called **overfitting** - the model learns the noise and specific patterns in training data that don't generalize

### The Solution: Train/Test Split

We divide our data into two parts:
- **Training Set** (typically 80%): Used to train the model
- **Test Set** (typically 20%): Used only for final evaluation - the model never sees this during training

This simulates real-world conditions where the model encounters new data.

## Understanding Overfitting and Underfitting

### Overfitting
- **What it is**: Model performs excellently on training data but poorly on test data
- **Cause**: Model is too complex, learns noise and specific details instead of general patterns
- **Signs**: Very high training accuracy, much lower test accuracy
- **Solution**: Simplify the model, add regularization, get more data

### Underfitting
- **What it is**: Model performs poorly on both training and test data
- **Cause**: Model is too simple, can't capture underlying patterns
- **Signs**: Low accuracy on both training and test sets
- **Solution**: More complex model, better features, feature engineering

### The Sweet Spot
We want a model that performs well on both training and test data - this indicates good generalization.

## Core Evaluation Metrics

### For Regression Problems

#### Mean Squared Error (MSE)
```
MSE = (1/n) * Σ(actual - predicted)²
```
- Measures average squared difference between predictions and actual values
- Lower is better
- Sensitive to outliers (squares amplify large errors)

#### Root Mean Squared Error (RMSE)
```
RMSE = √MSE
```
- Same as MSE but in original units (more interpretable)
- Lower is better

#### Mean Absolute Error (MAE)
```
MAE = (1/n) * Σ|actual - predicted|
```
- Measures average absolute difference
- Less sensitive to outliers than MSE
- Lower is better

#### R² Score (Coefficient of Determination)
```
R² = 1 - (SS_res / SS_tot)
```
- Measures proportion of variance explained by the model
- Range: 0 to 1 (can be negative for very poor models)
- Higher is better
- 1.0 = Perfect predictions
- 0.0 = Model predicts mean of target variable
- Negative = Worse than predicting the mean

### For Classification Problems

#### Accuracy
```
Accuracy = (Correct Predictions) / (Total Predictions)
```
- Percentage of correct predictions
- Good for balanced datasets
- Can be misleading for imbalanced datasets

#### Precision
```
Precision = TP / (TP + FP)
```
- Of all positive predictions, how many were actually positive?
- High precision = low false positive rate
- Important when false positives are costly

#### Recall (Sensitivity)
```
Recall = TP / (TP + FN)
```
- Of all actual positives, how many did we catch?
- High recall = low false negative rate
- Important when false negatives are costly

#### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
- Harmonic mean of precision and recall
- Balances both metrics
- Useful when you need both precision and recall to be good

#### Confusion Matrix
A table showing:
```
                 Predicted
              | Positive | Negative |
Actual  ------|----------|----------|
Positive      |    TP    |    FN    |
Negative      |    FP    |    TN    |
```
- TP: True Positives
- TN: True Negatives  
- FP: False Positives
- FN: False Negatives

## Cross-Validation

### The Problem with Single Split
One train/test split might be unlucky - maybe the test set happens to be particularly easy or hard.

### K-Fold Cross-Validation
1. Split data into K folds (typically 5 or 10)
2. Train on K-1 folds, test on 1 fold
3. Repeat K times, each fold gets to be the test set once
4. Average the results

**Benefits:**
- More robust evaluation
- Better use of data
- Reduces variance in performance estimates

## Best Practices

### 1. Always Split Your Data
Never evaluate on training data alone

### 2. Use Appropriate Metrics
- Regression: MSE, RMSE, MAE, R²
- Classification: Accuracy, Precision, Recall, F1-score, Confusion Matrix

### 3. Look for Overfitting Signs
Large gap between training and test performance

### 4. Use Cross-Validation
Especially for small datasets

### 5. Consider Your Domain
Choose metrics that align with business objectives:
- Medical diagnosis: High recall (catch all diseases)
- Spam filtering: High precision (avoid false alarms)
- Balanced approach: F1-score

## Code Examples

### Basic Train/Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Model Evaluation
```python
# Train model
model.fit(X_train, y_train)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score

train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)
```

### Cross-Validation
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Average CV Score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

## Next Steps

Tomorrow we'll apply everything learned so far to a real project - the Titanic Survival Prediction challenge. This will integrate data loading, preprocessing, model training, and proper evaluation in one complete workflow.
