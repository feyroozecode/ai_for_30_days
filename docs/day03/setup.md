# Day 3 Setup Guide

## Prerequisites

Ensure you have completed Day 1 and Day 2:
- Python virtual environment activated
- Required packages installed (`pip install -r requirements.txt`)
- Basic understanding of pandas, numpy, and scikit-learn

## Today's Goals

1. Understand train/test splitting concepts
2. Learn evaluation metrics for regression and classification
3. Detect and prevent overfitting
4. Implement cross-validation

## Files Created Today

- `src/day03/evaluation.py` - Core evaluation functions
- `docs/day03/concepts.md` - Detailed concepts explanation
- `docs/day03/setup.md` - This setup guide
- `tests/day03/test_evaluation.py` - Tests (created in next step)

## Running the Code

From the project root directory:

```bash
# Activate your virtual environment
source ai_bootcamp_env/bin/activate  # Linux/Mac
# or
ai_bootcamp_env\Scripts\activate     # Windows

# Run the evaluation module directly
python src/day03/evaluation.py
```

## Expected Output

When you run the module, you should see:
1. Regression example with training and test metrics
2. Classification example with training and test metrics
3. Overfitting detection analysis
4. Human-readable metric summaries

## Testing

Run the tests to verify everything works:

```bash
# Run Day 3 tests specifically
python -m pytest tests/day03/ -v

# Run all tests
python -m pytest tests/ -v
```

## Troubleshooting

### Import Errors
Make sure you're running from the project root directory, or Python can't find the src modules.

### Missing Packages
Install requirements:
```bash
pip install -r requirements.txt
```

### Test Failures
Check that:
1. All imports are correct
2. Functions return expected data types
3. Edge cases are handled properly

## Next Steps

After completing today's exercises:
1. Try modifying the examples with different datasets
2. Experiment with different test sizes (10%, 30%, 50%)
3. Try different cross-validation fold counts
4. Apply these concepts to the Titanic dataset (coming tomorrow!)
