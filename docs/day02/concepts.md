# Day 2: Your First Models - Linear & Logistic Regression

## What is a "Feature" and a "Target"?

In machine learning:
- **Features** (X): The input variables used to make predictions. These are the characteristics or attributes of your data.
- **Target** (y): The output variable you want to predict. This is what you're trying to learn from the features.

## Linear Regression

### Concept
Linear Regression is a supervised learning algorithm used for **regression** tasks - predicting continuous numerical values.

The algorithm finds the best-fitting straight line through your data points. The line is defined by:
- **Slope (coefficients)**: How much the target changes for each unit change in a feature
- **Intercept**: The value of the target when all features are zero

### Mathematical Formula
```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε
```
- y: predicted target
- β₀: intercept
- β₁, β₂, ..., βₙ: coefficients for each feature
- x₁, x₂, ..., xₙ: feature values
- ε: error term

### When to Use
- Predicting house prices based on size, location, etc.
- Forecasting sales based on advertising spend
- Estimating temperature based on various weather factors

### Example
```python
from sklearn.linear_model import LinearRegression

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Check model parameters
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

## Logistic Regression

### Concept
Logistic Regression is a supervised learning algorithm used for **classification** tasks - predicting discrete categories or classes.

Despite its name, it's used for classification, not regression. It predicts the probability that an instance belongs to a particular class.

The algorithm uses the logistic (sigmoid) function to transform the linear combination of features into a probability between 0 and 1.

### Mathematical Formula
```
P(y=1|X) = 1 / (1 + e^(-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ)))
```
- P(y=1|X): probability that the target is class 1 given the features
- The sigmoid function squashes the output to be between 0 and 1

### When to Use
- Email spam detection (spam/not spam)
- Customer churn prediction (will leave/will stay)
- Medical diagnosis (disease/no disease)
- Binary classification problems

### Example
```python
from sklearn.linear_model import LogisticRegression

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)  # Class predictions (0 or 1)
probabilities = model.predict_proba(X_test)  # Probability predictions
```

## Key Differences: Regression vs Classification

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|-------------------|
| **Task Type** | Regression | Classification |
| **Output** | Continuous values | Discrete classes/probabilities |
| **Example** | House price: $250,000 | Spam: Yes/No |
| **Loss Function** | Mean Squared Error | Log Loss |
| **Use Case** | Predicting quantities | Predicting categories |

## Scikit-Learn Implementation

Both models follow the same API pattern in scikit-learn:

1. **Import** the model class
2. **Instantiate** the model with parameters
3. **Fit** the model to training data: `model.fit(X_train, y_train)`
4. **Predict** on new data: `model.predict(X_test)`

### Common Parameters
- `random_state`: For reproducible results
- `fit_intercept`: Whether to include an intercept term (usually True)

## Evaluation Metrics

### For Regression (Linear)
- **Mean Squared Error (MSE)**: Average of squared differences between predictions and actual values
- **R² Score**: Proportion of variance explained by the model (0-1, higher is better)

### For Classification (Logistic)
- **Accuracy**: Fraction of correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Next Steps

These simple models form the foundation for more complex algorithms. Understanding linear and logistic regression will help you:
- Interpret model coefficients
- Understand feature importance
- Debug more complex models
- Apply regularization techniques

In the next days, we'll learn about model evaluation, train/test splits, and handling overfitting!