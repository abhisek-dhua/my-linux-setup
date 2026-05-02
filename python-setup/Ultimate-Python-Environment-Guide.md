# Ultimate Python Environment Guide for Ubuntu

## Overview

This guide provides a complete Python development environment setup that works alongside your existing Ubuntu Python3 installation. It includes version management, virtual environments, package management, and comprehensive cheat sheets for professional Python development.

---

## 🚀 Quick Start (5 Minutes)

### 1. Install pyenv (Python Version Manager)

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to shell configuration
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc

# Reload shell
source ~/.zshrc
```

### 2. Install Python Versions

```bash
# Install multiple Python versions
pyenv install 3.11.7
pyenv install 3.12.0
pyenv install 3.10.13

# Set default version
pyenv global 3.11.7
```

### 3. Test Setup

```bash
# Verify installation
python --version
pip --version
pyenv versions
```

---

## 📋 Complete Setup Guide

### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install build dependencies
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev curl git
```

### Step 1: Install pyenv

```bash
# Install pyenv
curl https://pyenv.run | bash

# Add to shell configuration (for zsh)
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc

# For bash users, use ~/.bashrc instead of ~/.zshrc

# Reload shell configuration
source ~/.zshrc
```

### Step 2: Install Python Versions

```bash
# Install stable Python versions
pyenv install 3.11.7  # Current stable
pyenv install 3.12.0  # Latest
pyenv install 3.10.13 # LTS
pyenv install 3.9.18  # Legacy support

# Set global default
pyenv global 3.11.7

# Verify installation
python --version
pip --version
```

### Step 3: Install Essential Tools

```bash
# Upgrade pip
pip install --upgrade pip

# Install development tools
pip install ipython jupyter notebook
pip install pytest pytest-cov
pip install black flake8 mypy
pip install pre-commit

# Install package management tools
pip install pipx
pipx ensurepath
pip install poetry
pip install pip-tools
```

---

## 🎯 Project-Specific Python Version Management

### Global vs Local vs Shell

```bash
# Global (system-wide default)
pyenv global 3.11.7

# Local (project-specific)
cd /path/to/project
pyenv local 3.12.0  # Creates .python-version file

# Shell (current session only)
pyenv shell 3.10.13
```

### Project Setup Examples

```bash
# Project A - Python 3.12
mkdir project_a
cd project_a
pyenv local 3.12.0
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn

# Project B - Python 3.11
mkdir project_b
cd project_b
pyenv local 3.11.7
python -m venv venv
source venv/bin/activate
pip install django djangorestframework

# Project C - Python 3.10 (legacy)
mkdir project_c
cd project_c
pyenv local 3.10.13
python -m venv venv
source venv/bin/activate
pip install tensorflow==2.10.0
```

---

## 🌍 Virtual Environment Management

### Basic Virtual Environments

```bash
# Create virtual environment
python -m venv myenv

# Activate
source myenv/bin/activate

# Deactivate
deactivate

# Remove
rm -rf myenv
```

### pyenv Virtual Environments

```bash
# Create virtual environment with specific Python version
pyenv virtualenv 3.11.7 myproject
pyenv virtualenv 3.12.0 myproject_latest

# Activate
pyenv activate myproject

# Set local virtual environment
pyenv local myproject

# List virtual environments
pyenv virtualenvs

# Remove virtual environment
pyenv uninstall myproject
```

### Advanced Virtual Environment Features

```bash
# Virtual environment with system packages
python -m venv --system-site-packages myenv

# Virtual environment without pip
python -m venv --without-pip myenv

# Virtual environment with custom prompt
python -m venv --prompt "MyProject" myenv

# Clone virtual environment
cp -r myenv myenv_clone
```

---

## 📦 Package Management

### pip Commands

```bash
# Install packages
pip install package_name
pip install package_name==1.2.3
pip install package_name>=1.2.0,<2.0.0

# Install from requirements
pip install -r requirements.txt

# Install in editable mode
pip install -e .

# Install with extras
pip install package_name[extra1,extra2]

# Uninstall
pip uninstall package_name

# List packages
pip list
pip list --outdated
pip show package_name

# Freeze requirements
pip freeze > requirements.txt
```

### pipx for Command-Line Tools

```bash
# Install command-line tools globally
pipx install black
pipx install flake8
pipx install mypy
pipx install pre-commit

# List installed tools
pipx list

# Upgrade tools
pipx upgrade-all

# Uninstall tools
pipx uninstall black
```

### Poetry (Modern Python Packaging)

```bash
# Install poetry
pip install poetry

# Initialize new project
poetry new myproject
cd myproject

# Add dependencies
poetry add fastapi uvicorn
poetry add --dev pytest black flake8

# Install dependencies
poetry install

# Run commands in virtual environment
poetry run python main.py
poetry run pytest

# Activate virtual environment
poetry shell
```

---

## 🛠️ Development Tools Setup

### Code Quality Tools

```bash
# Install code quality tools
pip install black flake8 mypy isort

# Configure black (create pyproject.toml)
echo '[tool.black]
line-length = 88
target-version = ['py311']
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
''' > pyproject.toml

# Configure flake8 (create .flake8)
echo '[flake8]
max-line-length = 88
extend-ignore = E203, W503
exclude = .git,__pycache__,.venv,venv,build,dist
' > .flake8
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.3.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
EOF

# Install hooks
pre-commit install
```

### Testing Setup

```bash
# Install testing tools
pip install pytest pytest-cov pytest-mock pytest-asyncio

# Create pytest configuration (pyproject.toml)
cat >> pyproject.toml << 'EOF'

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
]
EOF
```

---

## 📚 Essential Cheat Sheets

### Python Version Management (pyenv)

```bash
# List versions
pyenv versions
pyenv install --list

# Install versions
pyenv install 3.11.7
pyenv install 3.12.0

# Switch versions
pyenv global 3.11.7
pyenv local 3.12.0
pyenv shell 3.10.13

# Virtual environments
pyenv virtualenv 3.11.7 myproject
pyenv activate myproject
pyenv deactivate
pyenv virtualenvs
pyenv uninstall myproject

# Update pyenv
pyenv update
```

### Virtual Environment (venv)

```bash
# Create
python -m venv myenv
python3.11 -m venv myenv
python -m venv --system-site-packages myenv

# Activate
source myenv/bin/activate  # Linux/macOS
myenv\Scripts\activate     # Windows

# Deactivate
deactivate

# Check status
echo $VIRTUAL_ENV
which python
which pip

# Remove
rm -rf myenv
```

### Package Management (pip)

```bash
# Install
pip install package_name
pip install package_name==1.2.3
pip install -r requirements.txt
pip install -e .
pip install git+https://github.com/user/repo.git

# Uninstall
pip uninstall package_name
pip uninstall -r requirements.txt

# List
pip list
pip list --outdated
pip show package_name

# Freeze
pip freeze > requirements.txt
pip freeze --local > requirements.txt

# Upgrade
pip install --upgrade package_name
pip install --upgrade pip

# Cache
pip cache dir
pip cache list
pip cache purge
```

### Development Tools

```bash
# Code formatting
black .
black --check .
isort .

# Linting
flake8 .
mypy .

# Testing
pytest
pytest -v
pytest -k "test_name"
pytest --cov=src
pytest --cov-report=html

# Pre-commit
pre-commit run
pre-commit run --all-files
pre-commit install
```

### Git Integration

```bash
# Initialize with Python
git init
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".pytest_cache/" >> .gitignore
echo ".coverage" >> .gitignore

# Commit with pre-commit
git add .
git commit -m "Initial commit"
```

---

## 🧪 Testing Your Setup

### Test Script

Create a test script to verify your setup:

```bash
# Create test script
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify Python environment setup
"""

import sys
import subprocess
import importlib

def test_python_version():
    """Test Python version"""
    print(f"✅ Python version: {sys.version}")
    return True

def test_pip():
    """Test pip installation"""
    try:
        import pip
        print(f"✅ pip version: {pip.__version__}")
        return True
    except ImportError:
        print("❌ pip not found")
        return False

def test_packages():
    """Test essential packages"""
    packages = [
        'ipython', 'jupyter', 'pytest', 'black', 'flake8', 'mypy'
    ]

    results = []
    for package in packages:
        try:
            module = importlib.import_module(package)
            print(f"✅ {package}: {getattr(module, '__version__', 'installed')}")
            results.append(True)
        except ImportError:
            print(f"❌ {package}: not installed")
            results.append(False)

    return all(results)

def test_virtual_environment():
    """Test virtual environment creation"""
    try:
        import venv
        print("✅ venv module available")
        return True
    except ImportError:
        print("❌ venv module not available")
        return False

def test_pyenv():
    """Test pyenv functionality"""
    try:
        result = subprocess.run(['pyenv', '--version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pyenv: {result.stdout.strip()}")
            return True
        else:
            print("❌ pyenv not working")
            return False
    except FileNotFoundError:
        print("❌ pyenv not found")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Python Environment Setup")
    print("=" * 50)

    tests = [
        test_python_version,
        test_pip,
        test_packages,
        test_virtual_environment,
        test_pyenv
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            results.append(False)

    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"📊 Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Your Python environment is ready.")
    else:
        print("⚠️  Some tests failed. Check the setup guide.")

if __name__ == "__main__":
    main()
EOF

# Make executable and run
chmod +x test_setup.py
python test_setup.py
```

### Project Template Test

```bash
# Create test project
mkdir test_project
cd test_project

# Set Python version
pyenv local 3.11.7

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install test packages
pip install requests pytest black flake8

# Create test file
cat > test_example.py << 'EOF'
import requests

def test_requests():
    response = requests.get('https://httpbin.org/get')
    return response.status_code == 200

if __name__ == "__main__":
    if test_requests():
        print("✅ Requests test passed")
    else:
        print("❌ Requests test failed")
EOF

# Run test
python test_example.py

# Test virtual environment
echo "Virtual environment: $VIRTUAL_ENV"
which python
pip list

# Cleanup
deactivate
cd ..
rm -rf test_project
```

---

## 🔧 Troubleshooting

### Common Issues

#### pyenv not found

```bash
# Check if pyenv is in PATH
echo $PATH | grep pyenv

# Manually add to PATH
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
```

#### Virtual environment issues

```bash
# Reinstall virtual environment
pip install --upgrade virtualenv
python -m venv --clear myenv

# Fix permissions
chmod +x myenv/bin/activate
```

#### Package installation issues

```bash
# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Clear cache
pip cache purge

# Install with verbose output
pip install -v package_name
```

#### Python version conflicts

```bash
# Check current version
python --version
which python

# Reset pyenv
pyenv global 3.11.7
pyenv rehash
```

---

## 📁 Project Structure Examples

### Basic Python Project

```
myproject/
├── src/
│   └── myproject/
│       ├── __init__.py
│       └── main.py
├── tests/
│   ├── __init__.py
│   └── test_main.py
├── venv/
├── requirements.txt
├── pyproject.toml
├── .flake8
├── .gitignore
└── README.md
```

### Web Application Project

```
webapp/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   └── routes.py
├── tests/
│   ├── __init__.py
│   └── test_app.py
├── static/
├── templates/
├── venv/
├── requirements.txt
├── pyproject.toml
├── .env
├── .gitignore
└── README.md
```

### Data Science Project

```
datascience/
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   └── models.py
├── data/
│   ├── raw/
│   └── processed/
├── tests/
├── venv/
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## 🎯 Best Practices

### Version Management

- Use `pyenv local` for project-specific Python versions
- Keep a `.python-version` file in each project
- Use virtual environments for all projects

### Package Management

- Use `requirements.txt` for simple projects
- Use `pyproject.toml` for modern projects
- Use `poetry` for complex dependency management
- Pin package versions in production

### Development Workflow

- Use pre-commit hooks for code quality
- Write tests for all code
- Use type hints
- Follow PEP 8 style guide

### Security

- Keep packages updated
- Use virtual environments
- Don't install packages globally
- Use `pip check` to find security issues

---

## 📚 Additional Resources

### Documentation

- [pyenv Documentation](https://github.com/pyenv/pyenv)
- [Python Virtual Environments](https://docs.python.org/3/library/venv.html)
- [pip User Guide](https://pip.pypa.io/en/stable/user_guide/)
- [Poetry Documentation](https://python-poetry.org/docs/)

### Tools

- [Black Code Formatter](https://black.readthedocs.io/)
- [Flake8 Linter](https://flake8.pycqa.org/)
- [MyPy Type Checker](https://mypy.readthedocs.io/)
- [Pre-commit Hooks](https://pre-commit.com/)

---

**🎉 Congratulations! You now have a professional Python development environment with full version management, virtual environments, and all essential development tools.**
