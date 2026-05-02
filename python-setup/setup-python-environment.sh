#!/bin/bash

# Ultimate Python Environment Setup Script
# This script automates the complete Python environment setup on Ubuntu
# Based on the Ultimate Python Environment Guide

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect shell
detect_shell() {
    if [ -n "$ZSH_VERSION" ]; then
        echo "zsh"
    elif [ -n "$BASH_VERSION" ]; then
        echo "bash"
    else
        echo "unknown"
    fi
}

# Function to get shell config file
get_shell_config() {
    local shell_type=$(detect_shell)
    case $shell_type in
        "zsh")
            echo "$HOME/.zshrc"
            ;;
        "bash")
            echo "$HOME/.bashrc"
            ;;
        *)
            echo "$HOME/.bashrc"
            ;;
    esac
}

# Function to check if running as root
check_not_root() {
    if [ "$EUID" -eq 0 ]; then
        print_error "This script should not be run as root. Please run as a regular user."
        exit 1
    fi
}

# Function to update system packages
update_system() {
    print_header "Updating System Packages"
    print_step "Updating package list..."
    sudo apt update
    
    print_step "Upgrading packages..."
    sudo apt upgrade -y
    
    print_step "Installing essential build dependencies..."
    sudo apt install -y \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        python3-pip \
        python3-venv \
        git \
        unzip
}

# Function to install pyenv
install_pyenv() {
    print_header "Installing pyenv"
    
    if command_exists pyenv; then
        print_warning "pyenv is already installed. Skipping installation."
        return 0
    fi
    
    print_step "Installing pyenv..."
    curl https://pyenv.run | bash
    
    # Add pyenv to shell configuration
    local shell_config=$(get_shell_config)
    print_step "Configuring pyenv in $shell_config..."
    
    # Check if pyenv configuration already exists
    if ! grep -q "pyenv" "$shell_config"; then
        cat >> "$shell_config" << 'EOF'

# pyenv configuration
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
EOF
        print_status "pyenv configuration added to $shell_config"
    else
        print_warning "pyenv configuration already exists in $shell_config"
    fi
    
    # Source the configuration for current session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PATH"
    eval "$(pyenv init -)"
    
    print_status "pyenv installed successfully"
}

# Function to install Python versions
install_python_versions() {
    print_header "Installing Python Versions"
    
    # Define Python versions to install
    local python_versions=("3.12.0" "3.11.7" "3.10.12")
    
    for version in "${python_versions[@]}"; do
        print_step "Installing Python $version..."
        if pyenv versions | grep -q "$version"; then
            print_warning "Python $version is already installed. Skipping."
        else
            pyenv install "$version"
            print_status "Python $version installed successfully"
        fi
    done
    
    # Set global Python version to latest
    print_step "Setting global Python version to 3.12.0 (latest)..."
    pyenv global 3.12.0
    
    # Verify installation
    print_step "Verifying Python installation..."
    python --version
    pip --version
}

# Function to install pipx
install_pipx() {
    print_header "Installing pipx"
    
    if command_exists pipx; then
        print_warning "pipx is already installed. Skipping installation."
        return 0
    fi
    
    print_step "Installing pipx..."
    python -m pip install --user pipx
    python -m pipx ensurepath
    
    # Add pipx to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    print_status "pipx installed successfully"
}

# Function to install essential packages
install_essential_packages() {
    print_header "Installing Essential Development Packages"
    
    # Add pipx to PATH
    export PATH="$HOME/.local/bin:$PATH"
    
    print_step "Installing essential packages..."
    pip install --upgrade pip
    pip install \
        ipython \
        jupyter \
        pytest \
        black \
        flake8 \
        mypy \
        requests \
        numpy \
        pandas \
        matplotlib \
        seaborn \
        scikit-learn
    
    print_status "Essential packages installed successfully"
}

# Function to install optional packages
install_optional_packages() {
    print_header "Installing Optional Packages"
    
    print_step "Installing optional development tools..."
    
    # Install poetry if not exists
    if ! command_exists poetry; then
        print_step "Installing poetry..."
        curl -sSL https://install.python-poetry.org | python3 -
        export PATH="$HOME/.local/bin:$PATH"
    else
        print_warning "poetry is already installed. Skipping."
    fi
    
    # Install pip-tools
    pip install pip-tools
    
    # Install isort
    pip install isort
    
    # Install pre-commit
    pip install pre-commit
    
    print_status "Optional packages installed successfully"
}

# Function to configure development tools
configure_dev_tools() {
    print_header "Configuring Development Tools"
    
    # Create .flake8 configuration
    print_step "Creating .flake8 configuration..."
    cat > "$HOME/.flake8" << 'EOF'
[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,.venv,venv,.env
EOF
    
    # Create .black configuration
    print_step "Creating .black configuration..."
    cat > "$HOME/.black" << 'EOF'
[tool.black]
line-length = 88
target-version = ['py312']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.hg
  | \.mypy_cache
  | \.tox
  | \.venv
  | build
  | dist
)/
'''
EOF
    
    # Create .isort configuration
    print_step "Creating .isort configuration..."
    cat > "$HOME/.isort.cfg" << 'EOF'
[settings]
profile = black
multi_line_output = 3
line_length = 88
known_first_party = your_package_name
EOF
    
    print_status "Development tools configured successfully"
}

# Function to create project template
create_project_template() {
    print_header "Creating Project Template"
    
    local template_dir="$HOME/python_project_template"
    
    if [ -d "$template_dir" ]; then
        print_warning "Project template already exists. Skipping creation."
        return 0
    fi
    
    print_step "Creating project template directory..."
    mkdir -p "$template_dir"
    
    # Create .gitignore
    cat > "$template_dir/.gitignore" << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# PEP 582
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
EOF
    
    # Create README.md
    cat > "$template_dir/README.md" << 'EOF'
# Python Project Template

This is a Python project template with best practices and development tools configured.

## Setup

1. Set local Python version:
   ```bash
   pyenv local 3.12.0
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development

- **Format code**: `black .`
- **Lint code**: `flake8 .`
- **Type check**: `mypy .`
- **Run tests**: `pytest`

## Project Structure

```
project/
├── src/
│   └── your_package/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── requirements.txt
├── requirements-dev.txt
├── setup.py
├── README.md
└── .gitignore
```
EOF
    
    # Create setup.py template
    cat > "$template_dir/setup.py" << 'EOF'
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="your-package-name",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A short description of your package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your-package-name",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "isort>=5.0",
        ],
    },
)
EOF
    
    # Create requirements.txt template
    cat > "$template_dir/requirements.txt" << 'EOF'
# Core dependencies
requests>=2.25.0
numpy>=1.20.0
pandas>=1.3.0

# Development dependencies (install with: pip install -r requirements-dev.txt)
EOF
    
    # Create requirements-dev.txt template
    cat > "$template_dir/requirements-dev.txt" << 'EOF'
# Development dependencies
-r requirements.txt

# Testing
pytest>=6.0
pytest-cov>=2.10
pytest-mock>=3.6

# Code quality
black>=21.0
flake8>=3.8
mypy>=0.800
isort>=5.0

# Documentation
sphinx>=4.0
sphinx-rtd-theme>=0.5

# Pre-commit hooks
pre-commit>=2.15
EOF
    
    print_status "Project template created at $template_dir"
}

# Function to create test script
create_test_script() {
    print_header "Creating Test Script"
    
    local test_script="python_test_setup.py"
    
    if [ -f "$test_script" ]; then
        print_warning "Test script already exists. Skipping creation."
        return 0
    fi
    
    print_step "Creating comprehensive test script..."
    
    # Copy the test script content (simplified version)
    cat > "$test_script" << 'EOF'
#!/usr/bin/env python3
"""
Quick test script to verify Python environment setup
"""

import sys
import subprocess
import importlib

def test_python_version():
    print(f"✅ Python version: {sys.version}")
    return True

def test_pyenv():
    try:
        result = subprocess.run(['pyenv', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pyenv: {result.stdout.strip()}")
            return True
        else:
            print("❌ pyenv not working")
            return False
    except FileNotFoundError:
        print("❌ pyenv not found")
        return False

def test_essential_packages():
    packages = ['ipython', 'jupyter', 'pytest', 'black', 'flake8', 'mypy']
    results = []
    
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'installed')
            print(f"✅ {package}: {version}")
            results.append(True)
        except ImportError:
            print(f"❌ {package}: not installed")
            results.append(False)
    
    return all(results)

def test_virtual_environment():
    try:
        import venv
        print("✅ venv module available")
        return True
    except ImportError:
        print("❌ venv module not available")
        return False

def main():
    print("🧪 Quick Python Environment Test")
    print("=" * 40)
    
    tests = [
        test_python_version,
        test_pyenv,
        test_essential_packages,
        test_virtual_environment,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Python environment is ready.")
    else:
        print("⚠️  Some tests failed. Please check the setup.")

if __name__ == "__main__":
    main()
EOF
    
    chmod +x "$test_script"
    print_status "Test script created: $test_script"
}

# Function to display final instructions
display_final_instructions() {
    print_header "Setup Complete!"
    
    echo -e "${GREEN}🎉 Python environment setup completed successfully!${NC}"
    echo
    echo -e "${CYAN}Next steps:${NC}"
    echo "1. Restart your terminal or run: source $(get_shell_config)"
    echo "2. Test the setup: python python_test_setup.py"
    echo "3. Create a new project:"
    echo "   mkdir my_project && cd my_project"
    echo "   pyenv local 3.12.0"
    echo "   python -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install requests pytest black flake8"
    echo
    echo -e "${CYAN}Available Python versions:${NC}"
    pyenv versions
    echo
    echo -e "${CYAN}Project template available at:${NC}"
    echo "$HOME/python_project_template"
    echo
    echo -e "${CYAN}Useful commands:${NC}"
    echo "• pyenv versions          - List installed Python versions"
    echo "• pyenv global 3.12.0     - Set global Python version"
    echo "• pyenv local 3.12.0      - Set local Python version for project"
    echo "• python -m venv venv     - Create virtual environment"
    echo "• pip install package     - Install package"
    echo "• black .                 - Format code"
    echo "• flake8 .                - Lint code"
    echo "• pytest                  - Run tests"
    echo
    echo -e "${GREEN}Happy coding! 🐍${NC}"
}

# Main function
main() {
    print_header "Ultimate Python Environment Setup"
    echo "This script will set up a complete Python development environment"
    echo "with pyenv, multiple Python versions, and essential development tools."
    echo
    
    # Check if not running as root
    check_not_root
    
    # Confirm installation
    read -p "Do you want to continue with the installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Installation cancelled."
        exit 0
    fi
    
    # Run setup steps
    update_system
    install_pyenv
    install_python_versions
    install_pipx
    install_essential_packages
    install_optional_packages
    configure_dev_tools
    create_project_template
    create_test_script
    
    # Display final instructions
    display_final_instructions
}

# Run main function
main "$@" 