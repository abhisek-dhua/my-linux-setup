#!/usr/bin/env python3
"""
Comprehensive test script to verify Python environment setup
Covers all components from the Ultimate Python Environment Guide
"""

import sys
import subprocess
import importlib
import pkg_resources
import os
import tempfile
import shutil

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

def test_essential_packages():
    """Test essential development packages"""
    packages = [
        'ipython', 'jupyter', 'pytest', 'black', 'flake8', 'mypy'
    ]
    
    results = []
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
            results.append(True)
        except pkg_resources.DistributionNotFound:
            try:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'installed')
                print(f"✅ {package}: {version}")
                results.append(True)
            except ImportError:
                print(f"❌ {package}: not installed")
                results.append(False)
    
    return all(results)

def test_optional_packages():
    """Test optional but recommended packages"""
    packages = [
        'poetry', 'pip-tools', 'isort', 'pre-commit'
    ]
    
    results = []
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"✅ {package}: {version}")
            results.append(True)
        except pkg_resources.DistributionNotFound:
            print(f"⚠️  {package}: not installed (optional)")
            results.append(True)  # Not critical
    
    return True  # All optional packages are optional

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

def test_pyenv_versions():
    """Test pyenv Python versions"""
    try:
        result = subprocess.run(['pyenv', 'versions'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            versions = result.stdout.strip()
            print(f"✅ pyenv versions available: {len(versions.split())} versions")
            return True
        else:
            print("❌ pyenv versions not working")
            return False
    except FileNotFoundError:
        print("❌ pyenv not found")
        return False

def test_python_path():
    """Test Python path"""
    print(f"✅ Python executable: {sys.executable}")
    print(f"✅ Python path: {sys.path[0]}")
    return True

def test_pipx():
    """Test pipx installation"""
    try:
        result = subprocess.run(['pipx', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pipx: {result.stdout.strip()}")
            return True
        else:
            print("❌ pipx not working")
            return False
    except FileNotFoundError:
        print("❌ pipx not found")
        return False

def test_virtual_environment_creation():
    """Test actual virtual environment creation"""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        venv_path = os.path.join(temp_dir, 'test_venv')
        
        # Create virtual environment
        result = subprocess.run([sys.executable, '-m', 'venv', venv_path], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Test activation script exists
            if os.path.exists(os.path.join(venv_path, 'bin', 'activate')):
                print("✅ Virtual environment creation successful")
                shutil.rmtree(temp_dir)
                return True
            else:
                print("❌ Virtual environment activation script missing")
                shutil.rmtree(temp_dir)
                return False
        else:
            print(f"❌ Virtual environment creation failed: {result.stderr}")
            shutil.rmtree(temp_dir)
            return False
    except Exception as e:
        print(f"❌ Virtual environment test error: {e}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return False

def test_package_installation():
    """Test package installation in virtual environment"""
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        venv_path = os.path.join(temp_dir, 'test_venv')
        
        # Create virtual environment
        subprocess.run([sys.executable, '-m', 'venv', venv_path], 
                      capture_output=True, text=True)
        
        # Activate virtual environment and install package
        activate_script = os.path.join(venv_path, 'bin', 'activate')
        pip_path = os.path.join(venv_path, 'bin', 'pip')
        
        if os.path.exists(pip_path):
            # Install a test package
            result = subprocess.run([pip_path, 'install', 'requests'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Package installation in virtual environment successful")
                shutil.rmtree(temp_dir)
                return True
            else:
                print(f"❌ Package installation failed: {result.stderr}")
                shutil.rmtree(temp_dir)
                return False
        else:
            print("❌ pip not found in virtual environment")
            shutil.rmtree(temp_dir)
            return False
    except Exception as e:
        print(f"❌ Package installation test error: {e}")
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        return False

def test_development_tools():
    """Test development tools functionality"""
    tools_tests = []
    
    # Test black
    try:
        result = subprocess.run(['black', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ black: {result.stdout.strip()}")
            tools_tests.append(True)
        else:
            print("❌ black not working")
            tools_tests.append(False)
    except FileNotFoundError:
        print("❌ black not found")
        tools_tests.append(False)
    
    # Test flake8
    try:
        result = subprocess.run(['flake8', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ flake8: {result.stdout.strip()}")
            tools_tests.append(True)
        else:
            print("❌ flake8 not working")
            tools_tests.append(False)
    except FileNotFoundError:
        print("❌ flake8 not found")
        tools_tests.append(False)
    
    # Test pytest
    try:
        result = subprocess.run(['pytest', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pytest: {result.stdout.strip()}")
            tools_tests.append(True)
        else:
            print("❌ pytest not working")
            tools_tests.append(False)
    except FileNotFoundError:
        print("❌ pytest not found")
        tools_tests.append(False)
    
    return all(tools_tests)

def test_pyenv_virtualenv():
    """Test pyenv virtual environment functionality"""
    try:
        result = subprocess.run(['pyenv', 'virtualenvs'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ pyenv virtualenv command working")
            return True
        else:
            print("❌ pyenv virtualenv not working")
            return False
    except FileNotFoundError:
        print("❌ pyenv not found")
        return False

def test_poetry():
    """Test poetry installation and functionality"""
    try:
        result = subprocess.run(['poetry', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ poetry: {result.stdout.strip()}")
            return True
        else:
            print("❌ poetry not working")
            return False
    except FileNotFoundError:
        print("⚠️  poetry not installed (optional)")
        return True  # Optional

def test_pre_commit():
    """Test pre-commit installation"""
    try:
        result = subprocess.run(['pre-commit', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ pre-commit: {result.stdout.strip()}")
            return True
        else:
            print("❌ pre-commit not working")
            return False
    except FileNotFoundError:
        print("⚠️  pre-commit not installed (optional)")
        return True  # Optional

def test_system_integration():
    """Test system integration"""
    tests = []
    
    # Test PATH includes pyenv
    path = os.environ.get('PATH', '')
    if 'pyenv' in path:
        print("✅ pyenv in PATH")
        tests.append(True)
    else:
        print("❌ pyenv not in PATH")
        tests.append(False)
    
    # Test shell configuration
    home = os.path.expanduser('~')
    zshrc = os.path.join(home, '.zshrc')
    bashrc = os.path.join(home, '.bashrc')
    
    config_found = False
    if os.path.exists(zshrc):
        with open(zshrc, 'r') as f:
            if 'pyenv' in f.read():
                print("✅ pyenv configured in .zshrc")
                config_found = True
    
    if os.path.exists(bashrc) and not config_found:
        with open(bashrc, 'r') as f:
            if 'pyenv' in f.read():
                print("✅ pyenv configured in .bashrc")
                config_found = True
    
    if not config_found:
        print("⚠️  pyenv not found in shell configuration")
    
    tests.append(True)  # Not critical
    
    return all(tests)

def test_complete_project_workflow():
    """Test complete project creation and local environment workflow"""
    print("\n🚀 Testing Complete Project Workflow:")
    print("-" * 40)
    
    try:
        # Create temporary project directory
        temp_dir = tempfile.mkdtemp()
        project_dir = os.path.join(temp_dir, 'test_project')
        os.makedirs(project_dir)
        
        # Change to project directory
        original_dir = os.getcwd()
        os.chdir(project_dir)
        
        print(f"📁 Created project directory: {project_dir}")
        
        # Step 1: Set local Python version
        try:
            result = subprocess.run(['pyenv', 'local', '3.11.7'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Set local Python version: 3.11.7")
                
                # Check if .python-version file was created
                if os.path.exists('.python-version'):
                    with open('.python-version', 'r') as f:
                        version = f.read().strip()
                    print(f"✅ .python-version file created: {version}")
                else:
                    print("❌ .python-version file not created")
                    return False
            else:
                print(f"❌ Failed to set local Python version: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error setting local Python version: {e}")
            return False
        
        # Step 2: Create virtual environment
        try:
            result = subprocess.run([sys.executable, '-m', 'venv', 'venv'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Created virtual environment: venv/")
            else:
                print(f"❌ Failed to create virtual environment: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error creating virtual environment: {e}")
            return False
        
        # Step 3: Activate virtual environment and install packages
        try:
            # Get virtual environment Python and pip paths
            venv_python = os.path.join(project_dir, 'venv', 'bin', 'python')
            venv_pip = os.path.join(project_dir, 'venv', 'bin', 'pip')
            
            if not os.path.exists(venv_python):
                print("❌ Virtual environment Python not found")
                return False
            
            # Install test packages
            packages = ['requests', 'pytest', 'black', 'flake8']
            for package in packages:
                result = subprocess.run([venv_pip, 'install', package], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✅ Installed {package} in virtual environment")
                else:
                    print(f"❌ Failed to install {package}: {result.stderr}")
                    return False
        except Exception as e:
            print(f"❌ Error installing packages: {e}")
            return False
        
        # Step 4: Create a test Python file
        try:
            test_file = os.path.join(project_dir, 'test_app.py')
            with open(test_file, 'w') as f:
                f.write('''#!/usr/bin/env python3
"""
Test application for project workflow
"""

import requests

def test_requests():
    """Test requests functionality"""
    try:
        response = requests.get('https://httpbin.org/get')
        return response.status_code == 200
    except Exception:
        return False

def main():
    """Main function"""
    if test_requests():
        print("✅ Requests test passed")
        return True
    else:
        print("❌ Requests test failed")
        return False

if __name__ == "__main__":
    main()
''')
            print("✅ Created test application: test_app.py")
        except Exception as e:
            print(f"❌ Error creating test file: {e}")
            return False
        
        # Step 5: Test the application in virtual environment
        try:
            result = subprocess.run([venv_python, 'test_app.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0 and "✅ Requests test passed" in result.stdout:
                print("✅ Test application runs successfully in virtual environment")
            else:
                print(f"❌ Test application failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error running test application: {e}")
            return False
        
        # Step 6: Test development tools in virtual environment
        try:
            # Test black formatting
            result = subprocess.run([os.path.join(project_dir, 'venv', 'bin', 'black'), '--check', 'test_app.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Black formatting check passed")
            else:
                print("⚠️  Black formatting check failed (code needs formatting)")
            
            # Test flake8 linting
            result = subprocess.run([os.path.join(project_dir, 'venv', 'bin', 'flake8'), 'test_app.py'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Flake8 linting passed")
            else:
                print(f"⚠️  Flake8 linting issues: {result.stdout}")
            
            # Test pytest
            result = subprocess.run([os.path.join(project_dir, 'venv', 'bin', 'pytest'), '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Pytest available in virtual environment")
            else:
                print("❌ Pytest not working in virtual environment")
                return False
        except Exception as e:
            print(f"❌ Error testing development tools: {e}")
            return False
        
        # Step 7: Create requirements.txt
        try:
            result = subprocess.run([venv_pip, 'freeze'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                with open(os.path.join(project_dir, 'requirements.txt'), 'w') as f:
                    f.write(result.stdout)
                print("✅ Created requirements.txt")
            else:
                print("❌ Failed to create requirements.txt")
                return False
        except Exception as e:
            print(f"❌ Error creating requirements.txt: {e}")
            return False
        
        # Step 8: Test project structure
        try:
            project_files = os.listdir(project_dir)
            expected_files = ['venv', 'test_app.py', 'requirements.txt', '.python-version']
            missing_files = [f for f in expected_files if f not in project_files]
            
            if not missing_files:
                print("✅ Project structure complete")
                print(f"   📁 Project files: {', '.join(project_files)}")
            else:
                print(f"❌ Missing project files: {missing_files}")
                return False
        except Exception as e:
            print(f"❌ Error checking project structure: {e}")
            return False
        
        # Cleanup
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)
        
        print("✅ Complete project workflow test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Complete project workflow test error: {e}")
        # Cleanup on error
        try:
            if 'original_dir' in locals():
                os.chdir(original_dir)
            if 'temp_dir' in locals():
                shutil.rmtree(temp_dir)
        except:
            pass
        return False

def main():
    """Run all tests"""
    print("🧪 Comprehensive Python Environment Setup Test")
    print("=" * 60)
    
    # Core functionality tests
    print("\n📋 Core Functionality Tests:")
    print("-" * 30)
    core_tests = [
        test_python_version,
        test_pip,
        test_python_path,
        test_pyenv,
        test_pyenv_versions,
        test_pipx,
        test_virtual_environment,
    ]
    
    # Package tests
    print("\n📦 Package Tests:")
    print("-" * 30)
    package_tests = [
        test_essential_packages,
        test_optional_packages,
    ]
    
    # Virtual environment tests
    print("\n🌍 Virtual Environment Tests:")
    print("-" * 30)
    venv_tests = [
        test_virtual_environment_creation,
        test_package_installation,
        test_pyenv_virtualenv,
    ]
    
    # Development tools tests
    print("\n🛠️ Development Tools Tests:")
    print("-" * 30)
    dev_tests = [
        test_development_tools,
        test_poetry,
        test_pre_commit,
    ]
    
    # System integration tests
    print("\n🔧 System Integration Tests:")
    print("-" * 30)
    system_tests = [
        test_system_integration,
    ]
    
    # Complete project workflow test
    workflow_tests = [
        test_complete_project_workflow,
    ]
    
    # Run all tests
    all_tests = core_tests + package_tests + venv_tests + dev_tests + system_tests + workflow_tests
    
    results = []
    for test in all_tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ {test.__name__}: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your Python environment is fully ready.")
        print("✅ You can now work with multiple Python versions and projects.")
        print("✅ Complete project workflow is working perfectly!")
    elif passed >= total * 0.8:
        print("✅ Most tests passed! Your environment is ready for development.")
        print("⚠️  Some optional features may need attention.")
    else:
        print("⚠️  Several tests failed. Please check the setup guide.")
        print("🔧 You may need to reinstall or configure some components.")

if __name__ == "__main__":
    main() 