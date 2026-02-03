#!/bin/bash

# OpenCode AI Installer/Uninstaller Script
# Provides options to install or uninstall opencode-ai with cleanup

set -e

# Ensure npm and node are in PATH (especially for nvm users)
if [ -d "$HOME/.nvm" ]; then
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    [ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
fi

# Add common npm global paths to PATH
export PATH="$HOME/.nvm/versions/node/*/bin:$PATH"
export PATH="$HOME/.npm-global/bin:$PATH"

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}   OpenCode AI Manager${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_opencode_installed() {
    # Try multiple detection methods for better reliability
    
    # Method 1: command -v
    if command -v opencode &> /dev/null; then
        # Additional check: ensure the command actually works
        if opencode --version &> /dev/null || opencode --help &> /dev/null; then
            return 0
        fi
    fi
    
    # Method 2: Direct path search for nvm
    local nvm_opencode="$HOME/.nvm/versions/node/*/bin/opencode"
    for path in $nvm_opencode; do
        if [ -x "$path" ]; then
            return 0
        fi
    done
    
    # Method 3: Check npm global packages directly
    if command -v npm &> /dev/null; then
        if npm list -g opencode-ai &> /dev/null; then
            return 0
        fi
    fi
    
    return 1
}

install_opencode() {
    print_status "Installing OpenCode AI..."
    
    # Check if Node.js is installed
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install Node.js first."
        exit 1
    fi
    
    # Install globally
    npm install -g opencode-ai
    
    if [ $? -eq 0 ]; then
        print_status "OpenCode AI installed successfully!"
        echo "You can now run 'opencode' from your terminal."
    else
        print_error "Installation failed!"
        exit 1
    fi
}

uninstall_opencode() {
    print_status "Uninstalling OpenCode AI..."
    
    # Uninstall from npm
    if command -v npm &> /dev/null; then
        print_status "Removing npm global package..."
        npm uninstall -g opencode-ai || print_warning "Failed to uninstall via npm (may not be installed)"
    else
        print_warning "npm not found, skipping npm uninstall"
    fi
    
    # Remove local data and config directories
    print_status "Removing OpenCode data directories..."
    
    local dirs=(
        "$HOME/.local/share/OpenCode"
        "$HOME/.local/share/opencode" 
        "$HOME/.config/opencode"
        "$HOME/.cache/opencode"
        "$HOME/.cache/OpenCode"
    )
    
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_status "Removing: $dir"
            rm -rf "$dir"
        else
            print_warning "Directory not found: $dir"
        fi
    done
    
    # Clear npm cache (optional)
    read -p "Do you want to clear npm cache? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Clearing npm cache..."
        npm cache clean --force
    fi
    
    print_status "OpenCode AI uninstallation complete!"
}

clear_cache_only() {
    print_status "Clearing OpenCode AI cache and configuration..."
    
    # Remove cache and config directories only (keeping the application)
    local dirs=(
        "$HOME/.cache/opencode"
        "$HOME/.cache/OpenCode"
        "$HOME/.config/opencode"
    )
    
    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_status "Clearing: $dir"
            rm -rf "$dir"
        else
            print_warning "Directory not found: $dir"
        fi
    done
    
    print_status "Cache and configuration cleared successfully!"
}

update_opencode() {
    print_status "Updating OpenCode AI..."
    
    if ! check_opencode_installed; then
        print_error "OpenCode AI is not installed. Please install it first."
        return 1
    fi
    
    # Check if npm is available
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Cannot update."
        exit 1
    fi
    
    print_status "Updating to latest version..."
    npm update -g opencode-ai
    
    if [ $? -eq 0 ]; then
        print_status "OpenCode AI updated successfully!"
        print_status "New version: $(opencode --version 2>/dev/null || echo 'Unable to get version')"
    else
        print_error "Update failed!"
        exit 1
    fi
}

show_menu() {
    print_header
    
    if check_opencode_installed; then
        echo -e "${GREEN}OpenCode AI is currently installed${NC}"
    else
        echo -e "${RED}OpenCode AI is not installed${NC}"
    fi
    echo ""
    echo "What would you like to do?"
    echo "1) Install OpenCode AI"
    echo "2) Uninstall OpenCode AI"
    echo "3) Clear cache and config only"
    echo "4) Update OpenCode AI"
    echo "5) Check installation status"
    echo "6) Exit"
    echo ""
}

main() {
    while true; do
        show_menu
        read -p "Please enter your choice (1-6): " choice
        
        case $choice in
            1)
                if check_opencode_installed; then
                    print_warning "OpenCode AI is already installed!"
                    read -p "Do you want to reinstall it? (y/N): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        uninstall_opencode
                        install_opencode
                    fi
                else
                    install_opencode
                fi
                ;;
            2)
                if check_opencode_installed; then
                    uninstall_opencode
                else
                    print_warning "OpenCode AI is not installed!"
                fi
                ;;
            3)
                clear_cache_only
                ;;
            4)
                update_opencode
                ;;
            5)
                if check_opencode_installed; then
                    print_status "OpenCode AI is installed"
                    local version=$(opencode --version 2>/dev/null || npm list -g opencode-ai 2>/dev/null | grep opencode-ai | sed 's/.*@//' || echo 'Unable to get version')
                    print_status "Version: $version"
                    print_status "Location: $(which opencode 2>/dev/null || echo 'Not in PATH')"
                else
                    print_warning "OpenCode AI is not installed"
                    if command -v npm &> /dev/null; then
                        print_status "Checking npm global packages..."
                        npm list -g opencode-ai 2>/dev/null || print_status "opencode-ai not found in npm global packages"
                    fi
                fi
                ;;
            6)
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice! Please enter 1-6."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
        echo ""
    done
}

# Check if running with sudo (not recommended)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended for this script."
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

main