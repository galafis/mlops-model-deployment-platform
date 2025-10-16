# Examples

This directory contains example scripts demonstrating how to use the MLOps Model Deployment Platform.

## Available Examples

### 1. Simple Example (`simple_example.py`)

A basic demonstration of the core platform features:
- Initializing the deployment platform
- Creating and registering models
- Promoting models through stages
- Deploying models
- Making predictions
- Scaling deployments

**Run it:**
```bash
python examples/simple_example.py
```

### 2. Advanced Example (`../src/advanced_example.py`)

A comprehensive demonstration including:
- Synthetic data generation
- Model training with scikit-learn
- Model versioning
- Blue/Green deployment
- Real-time inference API
- Canary releases
- Traffic management

**Run it:**
```bash
python src/advanced_example.py
```

## Prerequisites

Make sure you have installed all dependencies:
```bash
pip install -r requirements.txt
```

## Learning Path

We recommend following this learning path:

1. Start with `simple_example.py` to understand the basic concepts
2. Move to `advanced_example.py` for a complete MLOps workflow
3. Explore the API documentation in the main README
4. Try building your own examples!

## Need Help?

- Check the main [README.md](../README.md) for detailed documentation
- Review [CONTRIBUTING.md](../CONTRIBUTING.md) for development guidelines
- Open an issue on GitHub for questions
