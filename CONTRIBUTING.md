# Contributing to MLOps Model Deployment Platform

Thank you for your interest in contributing to the MLOps Model Deployment Platform! This document provides guidelines and instructions for contributing to this project.

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment tool (venv, conda, etc.)

### Setting Up Development Environment

1. Fork the repository and clone your fork:
```bash
git clone https://github.com/YOUR_USERNAME/mlops-model-deployment-platform.git
cd mlops-model-deployment-platform
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install black flake8 mypy pytest-cov
```

4. Run tests to ensure everything is working:
```bash
pytest tests/ -v
```

## 📝 Development Workflow

### Branch Naming Convention

- Feature branches: `feature/your-feature-name`
- Bug fixes: `fix/bug-description`
- Documentation: `docs/what-you-document`
- Refactoring: `refactor/what-you-refactor`

### Making Changes

1. Create a new branch from `main`:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following our coding standards (see below)

3. Add tests for your changes

4. Run tests and linting:
```bash
# Run tests
pytest tests/ -v --cov=src

# Run Black formatter
black src/ tests/

# Run Flake8 linter
flake8 src/ tests/ --max-line-length=120
```

5. Commit your changes with descriptive commit messages:
```bash
git add .
git commit -m "feat: add new deployment strategy"
```

### Commit Message Guidelines

We follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

### Submitting Pull Requests

1. Push your changes to your fork:
```bash
git push origin feature/your-feature-name
```

2. Create a Pull Request on GitHub

3. Ensure all tests pass locally

4. Wait for code review and address any feedback

## 🎨 Coding Standards

### Python Style Guide

- Follow PEP 8 style guide
- Use Black for code formatting (120 character line length)
- Use type hints for function signatures
- Write docstrings for all public classes and functions

### Code Quality

- All new code must have tests
- Maintain or improve code coverage
- No linting errors (flake8)
- Follow existing code patterns

### Testing

- Write unit tests for all new functionality
- Write integration tests for API endpoints
- Test edge cases and error conditions
- Use descriptive test names

Example test structure:
```python
def test_feature_description():
    # Arrange
    setup_test_data()
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected_value
```

## 📚 Documentation

- Update README.md if adding new features
- Add docstrings to all public methods and classes
- Include code examples for complex features
- Update API documentation if changing endpoints

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining what the function does,
    important behaviors, or caveats.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: Description of when this is raised
    """
    pass
```

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, please include:

- Python version
- Operating system
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages (if any)
- Code samples (if applicable)

### Feature Requests

When requesting features, please:

- Explain the use case
- Describe the expected behavior
- Provide examples if possible
- Explain why this would be useful

## 💬 Getting Help

- Open an issue for questions
- Check existing issues before creating new ones
- Be respectful and constructive

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Thank You!

Your contributions help make this project better for everyone. We appreciate your time and effort!
