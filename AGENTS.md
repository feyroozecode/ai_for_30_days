# AI Bootcamp - Agent Guidelines

## Project Structure

This is a 30-day AI/ML bootcamp project with the following structure:

```
ai_for_30_days/
├── src/dayXX/          # Source code for each day
├── tests/dayXX/        # Tests for each day
├── docs/dayXX/         # Documentation for each day
│   ├── concepts.md     # English documentation
│   ├── concepts-fr.md  # French documentation
│   ├── setup.md        # Setup instructions (if applicable)
│   └── setup-fr.md     # French setup instructions
└── AGENTS.md           # This file
```

## Documentation Pattern Rule

**EVERY new day/module MUST include bilingual documentation following this exact pattern:**

### Required Files

For each new day (e.g., `day05`), create in `docs/day05/`:

1. **`concepts.md`** - Main concepts in English
2. **`concepts-fr.md`** - Main concepts in French (translation)

Optional (if setup is needed):
3. **`setup.md`** - Setup instructions in English
4. **`setup-fr.md`** - Setup instructions in French

### Documentation Template Structure

Each `concepts.md` file should follow this structure:

```markdown
# Day X: Title

## Main Concept Heading
Explanation of the concept...

### Sub-concepts
Details, examples...

## Another Main Concept
...

### Code Examples
\`\`\`python
from sklearn import something
# example code
\`\`\`

## Best Practices
- Point 1
- Point 2

## Next Steps
Brief teaser for the next day...
```

### Translation Guidelines for French

1. **Translate all headings** while keeping technical terms in English when standard:
   - "Hyperparameter Tuning" → "Optimisation des Hyperparamètres"
   - "Cross-Validation" → "Validation Croisée"
   - "Grid Search" → "Grid Search" (keep English if commonly used)

2. **Code examples remain in English** (Python code is universal)

3. **Keep URLs and references unchanged**

4. **Use formal French** ("vous" form, not "tu")

5. **Maintain the same structure** - same headings, same code blocks, same examples

### Implementation Checklist

When implementing a new day, verify:

- [ ] `src/dayXX/` directory created with `__init__.py` and module files
- [ ] `tests/dayXX/` directory created with `__init__.py` and test files
- [ ] `docs/dayXX/` directory created
- [ ] `docs/dayXX/concepts.md` written in English
- [ ] `docs/dayXX/concepts-fr.md` written in French
- [ ] All tests pass (`pytest tests/dayXX/`)
- [ ] Module can be run (`python src/dayXX/module.py`)

## Code Style Guidelines

### Python Code
- Use type hints where appropriate
- Follow PEP 8
- Include docstrings for all public functions
- Use `if __name__ == "__main__":` block for demo code

### Example Function Template

```python
def function_name(
    param1: np.ndarray,
    param2: Optional[str] = None,
    cv: int = 5
) -> Dict[str, Any]:
    """
    Brief description of what the function does.
    
    Args:
        param1 (np.ndarray): Description of param1
        param2 (str): Description of param2 (default: None)
        cv (int): Number of folds (default: 5)
        
    Returns:
        Dict: Description of return value
        
    Example:
        >>> result = function_name(X, y, cv=3)
        >>> print(result['score'])
    """
    # Implementation
    pass
```

### Test Style

Use pytest with descriptive test names:

```python
def test_function_name_basic():
    """Test basic functionality"""
    pass

def test_function_name_edge_case():
    """Test edge case handling"""
    pass
```

## Git Workflow

1. Implement code and tests
2. Run all tests: `pytest tests/`
3. Stage files: `git add src/dayXX/ tests/dayXX/ docs/dayXX/`
4. Do NOT commit unless explicitly asked by the user

## Prompt Response Pattern

When asked to implement a new day:

1. Review existing days to understand the progression
2. Check the roadmap.md for the day's objectives
3. Implement source code in `src/dayXX/`
4. Write comprehensive tests in `tests/dayXX/`
5. **Create bilingual docs in `docs/dayXX/`** (English + French)
6. Run all tests
7. Provide summary of what was implemented
