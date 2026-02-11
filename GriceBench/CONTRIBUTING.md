# Contributing to GriceBench

Thank you for your interest in contributing to GriceBench! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

This project follows a Code of Conduct to ensure a welcoming environment for all contributors. By participating, you agree to:

- Be respectful and inclusive
- Accept constructive criticism
- Focus on what is best for the community
- Show empathy towards others

## Getting Started

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/yourusername/GriceBench.git
cd GriceBench

# Add upstream remote
git remote add upstream https://github.com/originaluser/GriceBench.git
```

### Create a Branch

```bash
git checkout -b feature/your-feature-name
# Or for bug fixes:
git checkout -b fix/bug-description
```

---

## Development Setup

### 1. Install Dependencies

```bash
# Create virtual environment
python3.10 -m venv grice_env
source grice_env/bin/activate  # Windows: grice_env\\Scripts\\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 2. Download Development Data

```bash
# Download sample data for testing
python scripts/download_data.py --sample

# Or full dataset
python scripts/download_data.py --all
```

### 3. Verify Setup

```bash
# Run tests to ensure everything works
pytest tests/ -v

# Should see: All tests passed
```

---

## Coding Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line Length**: 100 characters (not 79)
- **Quotes**: Double quotes for strings, single for dict keys
- **Imports**: Organized with `isort`
- **Formatting**: Automated with `black`

### Code Formatting

We use automated formatters:

```bash
# Format code with black
black scripts/ tests/

# Sort imports
isort scripts/ tests/

# Lint with flake8
flake8 scripts/ tests/ --max-line-length=100

# Type checking with mypy (optional but encouraged)
mypy scripts/train_detector.py
```

### Pre-commit Hooks

Hooks automatically run before each commit:

```yaml
# .pre-commit-config.yaml
- black (formatting)
- isort (import sorting)
- flake8 (linting)
- trailing-whitespace
- end-of-file-fixer
```

### Documentation

- **Docstrings**: Use Google style
- **Type Hints**: Required for functions
- **Comments**: Explain "why", not "what"

**Example:**

```python
def detect_violations(
    context: str,
    response: str,
    threshold: float = 0.5
) -> tuple[dict[str, bool], dict[str, float]]:
    \"\"\"Detect maxim violations in a response.
    
    Args:
        context: Dialogue history
        response: Response to evaluate
        threshold: Probability threshold for positive classification
    
    Returns:
        Tuple of (violations dict, probability dict)
    
    Raises:
        ValueError: If context or response is empty
    \"\"\"
    pass
```

---

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests for individual functions
├── integration/    # Integration tests for components
└── end_to_end/     # Full pipeline tests
```

### Writing Tests

```python
# tests/unit/test_detector.py
import pytest
from scripts.detector import ViolationDetector

def test_detector_initialization():
    \"\"\"Test detector model loads correctly\"\"\"
    detector = ViolationDetector("microsoft/deberta-v3-base")
    assert detector is not None

def test_detector_forward_pass():
    \"\"\"Test detector produces valid outputs\"\"\"
    detector = ViolationDetector("microsoft/deberta-v3-base")
    # ... test code
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/unit/test_detector.py -v

# Run with coverage
pytest tests/ --cov=scripts --cov-report=html

# Run integration tests only
pytest tests/integration/ -v
```

### Test Coverage

We aim for **>80% test coverage**. Check coverage with:

```bash
pytest --cov=scripts --cov-report=term-missing
```

---

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**: `pytest tests/ -v`
2. **Format code**: `black scripts/ tests/`
3. **Lint code**: `flake8 scripts/ tests/`
4. **Update docs**: If adding features, update README.md
5. **Add tests**: New code must include tests

### PR Checklist

- [ ] Tests added and passing
- [ ] Code formatted with `black`
- [ ] Documentation updated
- [ ] No merge conflicts
- [ ] Descriptive commit messages
- [ ] PR description explains changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Related Issue
Fixes #123

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
```

### Review Process

1. **Automated checks** must pass (GitHub Actions)
2. **At least 1 reviewer** approval required
3. **Maintainer merge** after approval

---

## Reporting Issues

### Bug Reports

Use the **Bug Report** template:

```markdown
**Describe the bug**
Clear description of what's wrong

**To Reproduce**
Steps to reproduce:
1. Run command X
2. With parameters Y
3. See error Z

**Expected behavior**
What should happen

**Environment**
- OS: Ubuntu 22.04
- Python: 3.10.12
- PyTorch: 2.1.0
- GPU: NVIDIA T4

**Additional context**
Logs, screenshots, etc.
```

### Feature Requests

Use the **Feature Request** template:

```markdown
**Problem Statement**
What problem does this solve?

**Proposed Solution**
How would this feature work?

**Alternatives**
Other approaches considered

**Additional Context**
Why is this important?
```

---

## Development Workflow

### Typical Contribution Flow

```bash
# 1. Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/awesome-feature

# 3. Make changes
# ... edit files ...

# 4. Add tests
# ... write tests ...

# 5. Run checks
pytest tests/ -v
black scripts/ tests/
flake8 scripts/ tests/

# 6. Commit
git add .
git commit -m "feat: add awesome feature"

# 7. Push
git push origin feature/awesome-feature

# 8. Create PR on GitHub
# ... open pull request ...

# 9. Address review feedback
# ... make changes ...
git add .
git commit -m "fix: address review comments"
git push origin feature/awesome-feature
```

### Commit Message Guidelines

Follow **Conventional Commits**:

```
type(scope): subject

body (optional)

footer (optional)
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `ci`: CI/CD changes

**Examples:**
```
feat(detector): add calibration temperature scaling
fix(repair): handle empty context edge case
docs(readme): add installation troubleshooting
test(dpo): add preference pair filtering tests
```

---

## Questions?

- **Discussions**: Use [GitHub Discussions](https://github.com/yourusername/GriceBench/discussions)
- **Issues**: Report bugs or request features
- **Email**: your.email@university.edu

---

**Thank you for contributing to GriceBench!** 🎉
