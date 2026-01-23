# Day 2: Setup for Your First Models

## Overview

Day 2 introduces your first machine learning models using scikit-learn. We'll build Linear Regression and Logistic Regression models.

## Prerequisites

- Completed Day 1 setup
- Virtual environment activated
- All libraries from `requirements.txt` installed

## Required Libraries

Day 2 uses scikit-learn, which should already be installed from `requirements.txt`. If not, install it:

```bash
pip install scikit-learn
```

## Verify Installation

Run this Python code to verify scikit-learn is working:

```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.datasets import make_regression, make_classification

# Test Linear Regression
X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
model = LinearRegression()
model.fit(X, y)
print("Linear Regression coefficients:", model.coef_)
print("Linear Regression intercept:", model.intercept_)

# Test Logistic Regression
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
model = LogisticRegression()
model.fit(X, y)
print("Logistic Regression trained successfully!")

print("All Day 2 libraries working correctly!")
```

## Project Structure

Your project should now include:

```
ai_bootcamp_env/
├── src/
│   ├── day01/
│   │   ├── data_utils.py
│   │   └── __init__.py
│   └── day02/
│       ├── models.py
│       └── __init__.py
├── tests/
│   ├── day01/
│   │   ├── test_data_utils.py
│   │   └── __init__.py
│   └── day02/
│       ├── test_models.py
│       └── __init__.py
├── docs/
│   ├── day01/
│   └── day02/
│       ├── concepts.md
│       └── setup.md
└── requirements.txt
```

## Running the Code

### Test the Models

```bash
# Run the models directly
python src/day02/models.py
```

### Run Tests

```bash
# Run all tests
pytest

# Run only Day 2 tests
pytest tests/day02/
```

## Next Steps

Once setup is complete, you can:
1. Read the concepts in `docs/day02/concepts.md`
2. Run the example code in `src/day02/models.py`
3. Complete the exercises by modifying the code
4. Run tests to verify your implementations

Happy modeling!