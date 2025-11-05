# Developer Guides Documentation

This directory contains developer-focused documentation for mini-biai-1.

## Contents

- [Contributing Guide](contributing.md) - How to contribute to the project
- [Development Setup](development-setup.md) - Set up your development environment
- [Architecture Guide](architecture/) - Detailed architecture documentation
- [API Reference](api-reference.md) - Complete API reference

## For Contributors

Welcome to the development team! Here's what you need to know:

1. **[Contributing Guide](contributing.md)** - Guidelines and workflow
2. **[Development Setup](development-setup.md)** - Environment setup
3. **[Architecture Guide](architecture/overview.md)** - System design
4. **[API Reference](api-reference.md)** - Complete API documentation

## Development Workflow

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/mini-biai-1.git
cd mini-biai-1

# Setup development environment
pip install -e ".[dev]"
pre-commit install

# Create feature branch
git checkout -b feature/your-feature

# Make changes and test
pytest tests/
black src/ tests/

# Submit pull request
git push origin feature/your-feature
```

## Code Quality

We maintain high code quality through:

- **Automated Testing**: Comprehensive test suite
- **Code Formatting**: Black and isort
- **Type Checking**: MyPy for type safety
- **Linting**: Flake8 for style checking
- **Documentation**: All public APIs must be documented

---

For detailed documentation, see the individual files in this directory.