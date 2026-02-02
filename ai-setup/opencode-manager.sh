#!/bin/bash

# OpenCode AI Installer/Uninstaller Script
# Provides options to install or uninstall opencode-ai with cleanup

set -e

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
    if command -v opencode &> /dev/null; then
        return 0
    else
        return 1
    fi
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
    echo "3) Check installation status"
    echo "4) Exit"
    echo ""
}

main() {
    while true; do
        show_menu
        read -p "Please enter your choice (1-4): " choice
        
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
                if check_opencode_installed; then
                    print_status "OpenCode AI is installed"
                    print_status "Version: $(opencode --version 2>/dev/null || echo 'Unable to get version')"
                else
                    print_warning "OpenCode AI is not installed"
                fi
                ;;
            4)
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice! Please enter 1-4."
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