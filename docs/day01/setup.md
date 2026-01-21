# Day 1: Environment Setup

## Overview

This document guides you through setting up your development environment for the AI for 30 Days bootcamp.

## Prerequisites

- Python 3.8 or higher installed
- pip package manager
- Git

## Steps

### 1. Create a Virtual Environment

```bash
python -m venv ai_bootcamp_env
```

### 2. Activate the Virtual Environment

**On macOS/Linux:**
```bash
source ai_bootcamp_env/bin/activate
```

**On Windows:**
```bash
ai_bootcamp_env\Scripts\activate
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries installed successfully!")
```

### 5. Set Up Project Structure

The project follows this structure:
- `docs/`: Documentation
- `src/`: Source code
- `tests/`: Test files
- `README.md`: Project overview
- `requirements.txt`: Dependencies

## Next Steps

Once your environment is set up, proceed to learning the basic concepts in [concepts.md](concepts.md).