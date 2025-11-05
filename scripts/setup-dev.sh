#!/bin/bash

# BrainMod Step 2 - Development Environment Setup
# This script sets up the complete development environment for BrainMod Step 2

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    log_info "Checking Python version..."
    
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            log_success "Python version $PYTHON_VERSION is compatible"
            return 0
        else
            log_error "Python version $PYTHON_VERSION is not compatible. Requires Python 3.8+"
            return 1
        fi
    else
        log_error "Python 3 is not installed"
        return 1
    fi
}

# Function to check if pip is available
check_pip() {
    log_info "Checking pip availability..."
    
    if command_exists pip3; then
        log_success "pip3 is available"
        return 0
    else
        log_error "pip3 is not installed"
        return 1
    fi
}

# Function to create virtual environment
create_venv() {
    local venv_name="brainmod-env"
    
    if [ -d "$venv_name" ]; then
        log_warning "Virtual environment $venv_name already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing existing virtual environment..."
            rm -rf "$venv_name"
        else
            log_info "Using existing virtual environment"
            return 0
        fi
    fi
    
    log_info "Creating virtual environment: $venv_name"
    python3 -m venv "$venv_name"
    
    if [ $? -eq 0 ]; then
        log_success "Virtual environment created successfully"
    else
        log_error "Failed to create virtual environment"
        return 1
    fi
}

# Function to activate virtual environment
activate_venv() {
    local venv_name="brainmod-env"
    
    if [ -f "$venv_name/bin/activate" ]; then
        source "$venv_name/bin/activate"
        log_success "Virtual environment activated"
        return 0
    else
        log_error "Virtual environment activation script not found"
        return 1
    fi
}

# Function to install development dependencies
install_dev_deps() {
    log_info "Installing development dependencies..."
    
    # Upgrade pip first
    pip install --upgrade pip
    
    # Install development requirements
    if [ -f "requirements-dev.txt" ]; then
        pip install -r requirements-dev.txt
        log_success "Development dependencies installed from requirements-dev.txt"
    else
        log_warning "requirements-dev.txt not found, installing basic dev dependencies..."
        pip install pytest black flake8 mypy pre-commit
    fi
}

# Function to install project dependencies
install_project_deps() {
    log_info "Installing project dependencies..."
    
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        log_success "Project dependencies installed from requirements.txt"
    else
        log_warning "requirements.txt not found, installing basic dependencies..."
        pip install numpy scipy scikit-learn pandas matplotlib seaborn
    fi
}

# Function to setup pre-commit hooks
setup_precommit() {
    log_info "Setting up pre-commit hooks..."
    
    if [ -f ".pre-commit-config.yaml" ]; then
        pre-commit install
        log_success "Pre-commit hooks installed"
        
        # Run pre-commit on all files
        log_info "Running pre-commit checks..."
        pre-commit run --all-files
    else
        log_warning "No .pre-commit-config.yaml found, skipping pre-commit setup"
    fi
}

# Function to create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    directories=("logs" "data" "models" "results" "experiments")
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_success "Created directory: $dir"
        else
            log_info "Directory already exists: $dir"
        fi
    done
}

# Function to check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check available disk space (requires at least 2GB)
    available_space=$(df . | awk 'NR==2 {print $4}')
    required_space=$((2 * 1024 * 1024))  # 2GB in KB
    
    if [ "$available_space" -gt "$required_space" ]; then
        log_success "Sufficient disk space available"
    else
        log_warning "Low disk space. Consider freeing up space before running heavy computations."
    fi
    
    # Check available memory (requires at least 4GB)
    if command_exists free; then
        total_mem=$(free -m | awk 'NR==2{print $2}')
        if [ "$total_mem" -ge 4096 ]; then
            log_success "Sufficient memory available ($total_mem MB)"
        else
            log_warning "Low memory detected ($total_mem MB). Consider adding more RAM for optimal performance."
        fi
    fi
    
    # Check CPU cores
    cpu_cores=$(nproc)
    log_info "CPU cores available: $cpu_cores"
    
    if [ "$cpu_cores" -ge 4 ]; then
        log_success "Good number of CPU cores for parallel processing"
    else
        log_warning "Limited CPU cores may affect performance"
    fi
}

# Function to validate installation
validate_installation() {
    log_info "Validating installation..."
    
    # Test Python imports
    python3 -c "import numpy; import scipy; import sklearn; print('Core dependencies OK')"
    
    if [ $? -eq 0 ]; then
        log_success "Core dependencies validation passed"
    else
        log_error "Core dependencies validation failed"
        return 1
    fi
    
    # Test if brainmod modules can be imported
    if [ -d "brainmod" ]; then
        python3 -c "import brainmod; print('BrainMod modules OK')"
        
        if [ $? -eq 0 ]; then
            log_success "BrainMod modules validation passed"
        else
            log_warning "BrainMod modules not found or importable"
        fi
    fi
}

# Function to display setup summary
display_summary() {
    echo "
======================================"
    echo "    BrainMod Step 2 Setup Complete!"
    echo "======================================"
    echo ""
    echo "Setup Summary:"
    echo "- Python version: $(python3 --version)"
    echo "- Virtual environment: brainmod-env"
    echo "- Project directories created:"
    echo "  * logs/     - Log files"
    echo "  * data/     - Data files"
    echo "  * models/   - Trained models"
    echo "  * results/  - Experimental results"
    echo "  * experiments/ - Experiment artifacts"
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment: source brainmod-env/bin/activate"
    echo "2. Run a demo: ./quick_demo.sh"
    echo "3. Run advanced demos: ./advanced_demo.sh"
    echo ""
    echo "For development:"
    echo "- Run tests: pytest"
    echo "- Format code: black brainmod/"
    echo "- Check code: flake8 brainmod/"
    echo "- Setup complete! Happy coding!"
}

# Main setup function
main() {
    echo "======================================"
    echo "  BrainMod Step 2 Development Setup"
    echo "======================================"
    echo ""
    
    # Pre-flight checks
    check_python || exit 1
    check_pip || exit 1
    
    # System requirements
    check_system_requirements
    
    # Create virtual environment
    create_venv || exit 1
    
    # Activate virtual environment
    activate_venv || exit 1
    
    # Install dependencies
    install_project_deps || exit 1
    install_dev_deps || exit 1
    
    # Setup development tools
    setup_precommit
    
    # Create directories
    create_directories
    
    # Validate installation
    validate_installation
    
    # Display summary
    display_summary
    
    log_success "Setup completed successfully!"
}

# Run main function
main "$@"