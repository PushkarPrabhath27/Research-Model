# Contributing to GriceBench

Thank you for your interest in contributing to GriceBench! We welcome contributions from the community to help improve this framework for cooperative dialogue systems.

## 🤝 How to Contribute

### Reporting Bugs
If you find a bug, please create a GitHub issue with:
1. **Title**: Clear and descriptive.
2. **Description**: Steps to reproduce, expected behavior, and actual behavior.
3. **Environment**: Python version, OS, and relevant package versions.

### Suggesting Enhancements
We love new ideas! Please open an issue to discuss your proposal before starting work. This ensures your effort aligns with the project roadmap.

### Pull Requests
1. **Fork the repo** and create your branch from `main`.
2. **Install dependencies**: `pip install -r GriceBench/requirements-dev.txt` (or `requirements.txt`).
3. **Make your changes**: specific, focused, and well-tested.
4. **Run tests**: Ensure `pytest GriceBench/tests` passes.
5. **Lint your code**: We use `black` and `flake8`.
6. **Submit a Pull Request**: targeted at the `main` branch.

## 💻 Development & Code Style

- **Code Formatting**: We use [Black](https://github.com/psf/black) with default settings.
- **Type Hinting**: All new functions should have Python type hints.
- **Docstrings**: Use Google-style docstrings for all public functions and classes.

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Example function description.

    Args:
        param1 (int): The first parameter.
        param2 (str): The second parameter.

    Returns:
        bool: The return value.
    """
    return True
```

## 🧪 Testing

New features should include unit tests. We use `pytest`.
Run specific tests:
```bash
pytest GriceBench/tests/unit/test_detector.py
```

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License used by this project.
