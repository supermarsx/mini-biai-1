# Contributing to Mini-BIAI-1

Thank you for your interest in contributing to Mini-BIAI-1! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submission Process](#submission-process)

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Node.js 16+ (for documentation building)
- Docker (optional, for containerized development)

### Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/supermarsx/mini-biai-1.git
   cd mini-biai-1
   ```

2. **Install Dependencies**
   ```bash
   # Core dependencies
   pip install -r requirements.txt
   
   # Development dependencies
   pip install -e ".[dev]"
   
   # Pre-commit hooks
   pre-commit install
   ```

3. **Environment Setup**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your configuration
   ```

4. **Run Tests**
   ```bash
   # Run all tests
   python -m pytest
   
   # Run specific test module
   python -m pytest tests/test_coordinator.py
   ```

## Development Workflow

### Branch Strategy

- `main`: Stable release branch
- `develop`: Integration branch for features
- `feature/feature-name`: Feature development
- `bugfix/issue-description`: Bug fixes
- `hotfix/critical-fix`: Critical fixes for releases

### Feature Development

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Test**
   ```bash
   # Make your changes
   python -m pytest  # Run tests
   
   # Run code quality checks
   python -m black .
   python -m isort .
   python -m flake8 .
   python -m mypy .
   ```

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new memory optimization module"
   ```

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create PR on GitHub
   ```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- Line length: 88 characters (Black default)
- Quote style: Double quotes for strings
- Import organization: isort with custom configuration

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Style checking
- **mypy**: Type checking
- **pylint**: Code quality analysis

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Documentation Standards

All functions, classes, and modules must include:

- **Doxygen-style docstrings** with proper formatting
- **Parameter documentation** with types and descriptions
- **Return value documentation**
- **Usage examples** where appropriate
- **Cross-references** to related components

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests for individual modules
├── integration/    # Integration tests between components
├── performance/    # Performance and benchmark tests
├── conftest.py     # Pytest configuration and fixtures
└── utils/          # Test utilities and helpers
```

### Running Tests

```bash
# All tests
python -m pytest

# Specific test category
python -m pytest tests/unit/
python -m pytest tests/integration/

# With coverage
python -m pytest --cov=mini_biai_1 --cov-report=html

# Parallel execution
python -m pytest -n auto

# Failed tests only
python -m pytest --lf
```

### Coverage Requirements

- **Minimum Coverage**: 80% overall, 90% for critical modules
- **New Code**: Must include tests
- **Performance Critical**: Requires both unit and integration tests

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs build

# Serve locally
mkdocs serve
```

### Documentation Standards

- **Clear Structure**: Logical organization with navigation
- **Code Examples**: Practical, runnable examples
- **Cross-References**: Link related content
- **Screenshots**: Visual guides where helpful
- **Version Compatibility**: Note version requirements

## Submission Process

### Pull Request Guidelines

1. **PR Title**: Clear, descriptive title
   ```
   feat: add memory optimization module
   fix: resolve cache eviction issue
   docs: update API documentation
   ```

2. **PR Description**: Include:
   - What changed and why
   - How to test the changes
   - Breaking changes (if any)
   - Screenshots/examples (if applicable)

3. **Checklist**:
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] Changelog entry added
   - [ ] No breaking changes (or properly documented)

### Review Process

1. **Automated Checks**: CI/CD pipeline validation
2. **Code Review**: Maintainer review for style and logic
3. **Testing Review**: Verification of test coverage
4. **Documentation Review**: Check for completeness
5. **Final Approval**: Maintainer approval for merge

## Performance Guidelines

### Optimization Priorities

1. **Correctness First**: Ensure functionality before optimization
2. **Measurable Improvements**: Use benchmarks to validate improvements
3. **Memory Efficiency**: Consider memory usage in all changes
4. **Energy Efficiency**: Maintain low energy consumption targets

### Benchmarking

- **Performance Tests**: Required for performance-critical changes
- **Baseline Comparisons**: Compare against current benchmarks
- **Resource Usage**: Monitor memory and CPU usage
- **Energy Consumption**: Track power usage for SNN components

## Security

### Security Guidelines

- **Input Validation**: Validate all inputs
- **Secure Defaults**: Use secure default configurations
- **Dependency Updates**: Keep dependencies updated
- **Vulnerability Scanning**: Run security scans regularly

### Reporting Security Issues

See [SECURITY.md](SECURITY.md) for our security policy and reporting procedures.

## Getting Help

- **Discussions**: [GitHub Discussions](https://github.com/supermarsx/mini-biai-1/discussions)
- **Issues**: [GitHub Issues](https://github.com/supermarsx/mini-biai-1/issues)
- **Documentation**: [Project Documentation](https://mini-biai-1.readthedocs.io)
- **Community**: Join our community channels

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project website
- Academic publications (when applicable)

Thank you for contributing to Mini-BIAI-1! 🙏