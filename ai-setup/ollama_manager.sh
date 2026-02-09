#!/bin/bash

# Ollama Manager Script
# A comprehensive management tool for Ollama installation, updates, and maintenance

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${BLUE}=================================${NC}"
    echo -e "${BLUE}    OLLAMA MANAGER${NC}"
    echo -e "${BLUE}=================================${NC}"
    echo
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

# Check if Ollama is installed
check_ollama_installed() {
    if command -v ollama >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Get Ollama version
get_ollama_version() {
    if check_ollama_installed; then
        ollama --version 2>/dev/null | cut -d' ' -f2 || echo "unknown"
    else
        echo "not installed"
    fi
}

# Check if systemd service exists
check_service_exists() {
    if [ -f /etc/systemd/system/ollama.service ] || [ -f ~/.config/systemd/user/ollama.service ]; then
        return 0
    else
        return 1
    fi
}

# Install Ollama
install_ollama() {
    print_status "Installing Ollama..."
    
    if check_ollama_installed; then
        print_warning "Ollama is already installed. Version: $(get_ollama_version)"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi
    
    # Download and install Ollama
    print_status "Downloading Ollama..."
    if ! curl -fsSL https://ollama.com/install.sh | sh; then
        print_error "Failed to install Ollama"
        return 1
    fi
    
    print_status "Creating systemd service..."
    create_systemd_service
    
    print_status "Ollama installed successfully!"
    print_status "Version: $(get_ollama_version)"
}

# Create systemd service
create_systemd_service() {
    local current_user=$(whoami)
    local service_content="[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/local/bin/ollama serve
User=$current_user
Group=$current_user
Restart=always
RestartSec=3
Environment=\"OLLAMA_HOST=http://127.0.0.1:11434\"

[Install]
WantedBy=default.target"
    
    # Try to create system service first
    if echo "$service_content" | sudo tee /etc/systemd/system/ollama.service >/dev/null 2>&1; then
        print_status "Created system service"
        sudo systemctl daemon-reload >/dev/null 2>&1
        sudo systemctl enable ollama >/dev/null 2>&1
        sudo systemctl start ollama >/dev/null 2>&1
    else
        print_warning "Could not create system service, creating user service instead"
        mkdir -p ~/.config/systemd/user
        echo "$service_content" > ~/.config/systemd/user/ollama.service
        systemctl --user daemon-reload >/dev/null 2>&1
        systemctl --user enable ollama >/dev/null 2>&1
        systemctl --user start ollama >/dev/null 2>&1
    fi
}

# Uninstall Ollama
uninstall_ollama() {
    if ! check_ollama_installed; then
        print_warning "Ollama is not installed"
        return
    fi
    
    print_warning "This will completely remove Ollama and all models"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi
    
    print_status "Stopping Ollama service..."
    sudo systemctl stop ollama 2>/dev/null || true
    sudo systemctl disable ollama 2>/dev/null || true
    systemctl --user stop ollama 2>/dev/null || true
    systemctl --user disable ollama 2>/dev/null || true
    
    print_status "Removing systemd service..."
    sudo rm -f /etc/systemd/system/ollama.service
    rm -f ~/.config/systemd/user/ollama.service
    sudo systemctl daemon-reload 2>/dev/null || true
    systemctl --user daemon-reload 2>/dev/null || true
    
    print_status "Removing Ollama binary..."
    sudo rm -f /usr/local/bin/ollama
    sudo rm -rf /usr/local/share/ollama
    
    print_status "Removing Ollama data and models..."
    rm -rf ~/.ollama
    
    print_status "Ollama uninstalled successfully"
}

# Update Ollama
update_ollama() {
    if ! check_ollama_installed; then
        print_error "Ollama is not installed. Please install it first."
        return 1
    fi
    
    local current_version=$(get_ollama_version)
    print_status "Current version: $current_version"
    print_status "Updating Ollama..."
    
    # Stop service during update
    sudo systemctl stop ollama 2>/dev/null || true
    systemctl --user stop ollama 2>/dev/null || true
    
    # Re-run install script which handles updates
    if curl -fsSL https://ollama.com/install.sh | sh; then
        print_status "Restarting Ollama service..."
        sudo systemctl start ollama 2>/dev/null || systemctl --user start ollama 2>/dev/null || true
        print_status "Ollama updated successfully!"
        print_status "New version: $(get_ollama_version)"
    else
        print_error "Failed to update Ollama"
        return 1
    fi
}

# List installed models
list_models() {
    if ! check_ollama_installed; then
        print_error "Ollama is not installed"
        return 1
    fi
    
    print_status "Installed models:"
    ollama list 2>/dev/null || print_warning "Could not list models"
}

# Pull a model
pull_model() {
    if ! check_ollama_installed; then
        print_error "Ollama is not installed"
        return 1
    fi
    
    echo "Available popular models:"
    echo "1. llama3.2 (3B) - Lightweight, fast"
    echo "2. llama3.2 (1B) - Very lightweight"
    echo "3. qwen2.5 (7B) - Good balance"
    echo "4. codellama (7B) - Code focused"
    echo "5. Custom model name"
    echo
    
    read -p "Enter model name or choice (1-5): " choice
    
    case $choice in
        1) model="llama3.2:3b" ;;
        2) model="llama3.2:1b" ;;
        3) model="qwen2.5:7b" ;;
        4) model="codellama:7b" ;;
        5) 
            read -p "Enter custom model name: " model
            ;;
        *) 
            model=$choice
            ;;
    esac
    
    if [ -n "$model" ]; then
        print_status "Pulling model: $model"
        ollama pull "$model"
        print_status "Model pulled successfully!"
    fi
}

# Service management
manage_service() {
    echo "Service Management:"
    echo "1. Start service"
    echo "2. Stop service"
    echo "3. Restart service"
    echo "4. Check status"
    echo
    
    read -p "Enter choice (1-4): " choice
    
    case $choice in
        1)
            sudo systemctl start ollama 2>/dev/null || systemctl --user start ollama
            print_status "Service started"
            ;;
        2)
            sudo systemctl stop ollama 2>/dev/null || systemctl --user stop ollama
            print_status "Service stopped"
            ;;
        3)
            sudo systemctl restart ollama 2>/dev/null || systemctl --user restart ollama
            print_status "Service restarted"
            ;;
        4)
            echo "System service status:"
            sudo systemctl status ollama 2>/dev/null || echo "No system service"
            echo
            echo "User service status:"
            systemctl --user status ollama 2>/dev/null || echo "No user service"
            ;;
    esac
}

# Show main menu
show_menu() {
    clear
    print_header
    
    echo -e "${BLUE}Current Status:${NC}"
    echo -e "Ollama: $(check_ollama_installed && echo -e "${GREEN}Installed ($(get_ollama_version))${NC}" || echo -e "${RED}Not Installed${NC}")"
    echo -e "Service: $(check_service_exists && echo -e "${GREEN}Configured${NC}" || echo -e "${RED}Not Configured${NC}")"
    echo
    
    echo -e "${YELLOW}Options:${NC}"
    echo "1. Install Ollama"
    echo "2. Uninstall Ollama"
    echo "3. Update Ollama"
    echo "4. List Installed Models"
    echo "5. Pull/Download Model"
    echo "6. Service Management"
    echo "7. Exit"
    echo
}

# Main function
main() {
    while true; do
        show_menu
        read -p "Enter your choice (1-7): " choice
        
        case $choice in
            1)
                install_ollama
                ;;
            2)
                uninstall_ollama
                ;;
            3)
                update_ollama
                ;;
            4)
                list_models
                ;;
            5)
                pull_model
                ;;
            6)
                manage_service
                ;;
            7)
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please enter 1-7."
                ;;
        esac
        
        echo
        read -p "Press Enter to continue..."
    done
}

# Run main function
main "$@"