#!/bin/bash
# Development Environment Setup Script for BrainMod
# This script sets up a complete development environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    print_status "Checking Python installation..."
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
        print_success "Python 3 found: version $PYTHON_VERSION"
        
        # Check if version is >= 3.8
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python version is compatible (>= 3.8)"
        else
            print_error "Python 3.8 or higher is required"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Check if pip is available
check_pip() {
    print_status "Checking pip installation..."
    if command -v pip3 &> /dev/null; then
        print_success "pip3 found"
    elif command -v pip &> /dev/null; then
        print_success "pip found"
    else
        print_error "pip is not installed"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Setting up virtual environment..."
    
    if [ -d "venv" ]; then
        print_warning "Virtual environment 'venv' already exists"
        read -p "Do you want to remove it and create a new one? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
        else
            print_status "Using existing virtual environment"
            return
        fi
    fi
    
    python3 -m venv venv
    print_success "Virtual environment created"
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
}

# Upgrade pip and setuptools
upgrade_tools() {
    print_status "Upgrading pip and setuptools..."
    pip install --upgrade pip setuptools wheel
    print_success "Tools upgraded"
}

# Install development dependencies
install_dev_deps() {
    print_status "Installing development dependencies..."
    
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        print_success "Development dependencies installed"
    else
        print_warning "requirements-dev.txt not found, installing from pyproject.toml"
        pip install -e ".[dev,docs]"
        print_success "Dependencies installed from pyproject.toml"
    fi
}

# Install pre-commit hooks
setup_precommit() {
    print_status "Setting up pre-commit hooks..."
    if command -v pre-commit &> /dev/null; then
        pre-commit install
        print_success "Pre-commit hooks installed"
    else
        print_warning "pre-commit not found, skipping hook installation"
    fi
}

# Run initial linting
run_initial_lint() {
    print_status "Running initial code quality checks..."
    
    # Format code
    if command -v black &> /dev/null; then
        black src/ tests/ --check --diff
        print_success "Code formatting check passed"
    fi
    
    # Check imports
    if command -v isort &> /dev/null; then
        isort src/ tests/ --check-only --diff
        print_success "Import order check passed"
    fi
}

# Run tests
run_tests() {
    print_status "Running test suite..."
    
    if command -v pytest &> /dev/null; then
        pytest tests/ -v --tb=short
        print_success "Test suite passed"
    else
        print_warning "pytest not found, skipping tests"
    fi
}

# Setup complete
setup_complete() {
    print_success "Development environment setup complete!"
    echo
    echo -e "${GREEN}Next steps:${NC}"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Install BrainMod in development mode: make install-dev"
    echo "3. Run tests: make test"
    echo "4. Start coding!"
    echo
    echo -e "${BLUE}Available commands:${NC}"
    echo "  make test          - Run all tests"
    echo "  make test-coverage - Run tests with coverage"
    echo "  make lint          - Run linting"
    echo "  make format        - Format code"
    echo "  make docs          - Build documentation"
    echo "  make help          - Show all available commands"
}

# Main setup function
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}BrainMod Development Environment Setup${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo
    
    check_python
    check_pip
    create_venv
    activate_venv
    upgrade_tools
    install_dev_deps
    setup_precommit
    run_initial_lint
    # run_tests  # Optional: uncomment if you want to run tests during setup
    setup_complete
}

# Run main function
main "$@"