#!/bin/bash

# OpenCode AI Installer/Uninstaller Script
# Supports OpenCode v2 (current) - CLI, Desktop App
# Provides options to install or uninstall opencode with cleanup

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
export PATH="$HOME/.opencode/bin:$PATH"

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

check_official_installed() {
    # Check if installed via official script (~/.opencode/bin/)
    if [ -x "$HOME/.opencode/bin/opencode" ]; then
        return 0
    fi
    return 1
}

check_npm_installed() {
    # Check if installed via npm
    if command -v npm &> /dev/null; then
        if npm list -g @opencode/cli &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

check_desktop_installed() {
    # Check if desktop app is installed (AppImage in /opt/)
    if [ -f "/opt/opencode-desktop.AppImage" ] || [ -f "/opt/opencode.appimage" ]; then
        return 0
    fi
    # Check for .deb installation (package name is "opencode")
    if dpkg -l | grep -q "^ii.*opencode[[:space:]]"; then
        return 0
    fi
    return 1
}

check_opencode_installed() {
    # Check if any version is installed
    if check_official_installed || check_npm_installed || check_desktop_installed; then
        return 0
    fi
    # Also check PATH for any opencode
    if command -v opencode &> /dev/null; then
        if opencode --version &> /dev/null || opencode --help &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

get_official_version() {
    if [ -x "$HOME/.opencode/bin/opencode" ]; then
        "$HOME/.opencode/bin/opencode" --version 2>/dev/null || echo "?"
    else
        echo "?"
    fi
}

get_npm_version() {
    if command -v npm &> /dev/null; then
        npm list -g @opencode/cli 2>/dev/null | grep @opencode/cli | sed 's/.*@//' || echo "?"
    else
        echo "?"
    fi
}

get_desktop_version() {
    # Try to get version from AppImage
    if [ -f "/opt/opencode-desktop.AppImage" ]; then
        /opt/opencode-desktop.AppImage --version 2>/dev/null | head -1 || echo "?"
    elif [ -f "/opt/opencode.appimage" ]; then
        /opt/opencode.appimage --version 2>/dev/null | head -1 || echo "?"
    elif dpkg -l | grep -q "^ii.*opencode[[:space:]]"; then
        # Get version from dpkg
        dpkg -l | grep "^ii.*opencode[[:space:]]" | awk '{print $3}' | head -1
    else
        echo "?"
    fi
}

get_opencode_version() {
    opencode --version 2>/dev/null || echo "unknown"
}

get_opencode_location() {
    which opencode 2>/dev/null || echo "Not in PATH"
}

# Get latest version from GitHub releases
get_latest_version() {
    curl -fsSL --max-time 10 "https://api.github.com/repos/anomalyco/opencode/releases/latest" 2>/dev/null | grep -o '"tag_name": *"[^"]*"' | sed 's/.*"v\([^"]*\)".*/\1/' || echo "unknown"
}

install_official() {
    print_status "Installing OpenCode v2 via official install script..."
    
    # Check if curl is available
    if ! command -v curl &> /dev/null; then
        print_error "curl is required. Install with: sudo apt install curl"
        return 1
    fi
    
    if check_official_installed; then
        print_warning "OpenCode (official) is already installed!"
        read -p "Reinstall? (y/N): " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then return; fi
    fi
    
    # Use official install script
    curl -fsSL https://opencode.ai/v2/install | bash
    
    if [ $? -eq 0 ] && [ -x "$HOME/.opencode/bin/opencode" ]; then
        print_status "OpenCode (official) installed successfully!"
        echo "You can now run 'opencode' from your terminal."
        echo "Note: You may need to restart your terminal or run 'source ~/.bashrc' to update PATH."
    else
        print_error "Installation failed!"
        return 1
    fi
}

install_npm() {
    print_status "Installing OpenCode v2 via npm (@opencode/cli)..."
    
    # Check if Node.js is installed
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install Node.js first."
        return 1
    fi
    
    if check_npm_installed; then
        print_warning "OpenCode (npm) is already installed!"
        read -p "Reinstall? (y/N): " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then return; fi
    fi
    
    # Install globally
    npm install -g @opencode/cli
    
    if [ $? -eq 0 ]; then
        print_status "OpenCode (npm) installed successfully!"
        echo "You can now run 'opencode' from your terminal."
    else
        print_error "Installation failed!"
        return 1
    fi
}

install_desktop() {
    print_status "Installing OpenCode Desktop App..."
    
    for cmd in curl; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "$cmd is required. Install with: sudo apt install $cmd"
            return 1
        fi
    done
    
    if check_desktop_installed; then
        print_warning "OpenCode Desktop is already installed!"
        read -p "Reinstall? (y/N): " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then return; fi
    fi
    
    # Detect architecture
    case "$(uname -m)" in
        x86_64|amd64) arch="x64" ;;
        aarch64|arm64) arch="arm64" ;;
        *) 
            print_error "Unsupported architecture: $(uname -m)"
            return 1
            ;;
    esac
    
    # Download .deb file
    local url="https://opencode.ai/download/stable/linux-${arch}-deb"
    local deb_file="/tmp/opencode-desktop_${arch}.deb"
    
    print_status "Downloading OpenCode Desktop (.deb) for ${arch}..."
    if ! curl -fsSL --retry 3 -o "$deb_file" "$url"; then
        print_error "Failed to download .deb file"
        return 1
    fi
    
    print_status "Installing .deb package..."
    if ! sudo apt install -y "$deb_file"; then
        print_error "Failed to install .deb package"
        return 1
    fi
    
    # Clean up
    rm -f "$deb_file"
    
    if check_desktop_installed; then
        print_status "OpenCode Desktop installed successfully!"
        echo "You can now run it from your applications menu."
    else
        print_error "Installation verification failed!"
        return 1
    fi
}

uninstall_official() {
    print_status "Uninstalling OpenCode (official)..."
    
    # Remove v2 installation directory
    if [ -d "$HOME/.opencode" ]; then
        print_status "Removing OpenCode installation directory..."
        rm -rf "$HOME/.opencode"
    fi
    
    print_status "OpenCode (official) uninstalled!"
}

uninstall_npm() {
    print_status "Uninstalling OpenCode (npm)..."
    
    if command -v npm &> /dev/null; then
        print_status "Removing npm global package (@opencode/cli)..."
        npm uninstall -g @opencode/cli || print_warning "Failed to uninstall @opencode/cli via npm (may not be installed)"
        
        # Also try legacy package name
        npm uninstall -g @opencode-ai/cli 2>/dev/null || true
    else
        print_warning "npm not found, skipping npm uninstall"
    fi
    
    print_status "OpenCode (npm) uninstalled!"
}

uninstall_desktop() {
    print_status "Uninstalling OpenCode Desktop..."
    
    # Try apt remove first (package name is "opencode")
    if dpkg -l | grep -q "^ii.*opencode[[:space:]]"; then
        sudo apt remove -y opencode
        print_status "Removed via apt"
    fi
    
    # Remove AppImage if exists
    sudo rm -f /opt/opencode-desktop.AppImage /opt/opencode.appimage
    
    # Remove desktop entry
    sudo rm -f /usr/share/applications/opencode.desktop
    sudo rm -f /usr/share/applications/opencode-desktop.desktop
    
    # Update desktop database
    sudo update-desktop-database /usr/share/applications 2>/dev/null || true
    
    print_status "OpenCode Desktop uninstalled!"
}

uninstall_all_opencode() {
    print_warning "This will remove ALL OpenCode components!"
    read -p "Are you sure? (y/N): " -n 1 -r; echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uninstall_official
        uninstall_npm
        uninstall_desktop
        
        # Remove local data and config directories
        print_status "Removing OpenCode data directories..."
        
        local dirs=(
            "$HOME/.local/share/OpenCode"
            "$HOME/.local/share/opencode" 
            "$HOME/.config/opencode"
            "$HOME/.cache/opencode"
            "$HOME/.cache/OpenCode"
            "$HOME/.opencode"
        )
        
        for dir in "${dirs[@]}"; do
            if [ -d "$dir" ]; then
                print_status "Removing: $dir"
                rm -rf "$dir"
            fi
        done
        
        # Clear npm cache (optional)
        read -p "Do you want to clear npm cache? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Clearing npm cache..."
            npm cache clean --force
        fi
        
        print_status "All OpenCode components removed."
    else
        print_status "Cancelled."
    fi
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

update_official() {
    if check_official_installed; then
        print_status "Updating OpenCode (official)..."
        curl -fsSL https://opencode.ai/v2/install | bash
        if [ $? -eq 0 ]; then
            print_status "OpenCode (official) updated successfully!"
            print_status "New version: $(get_official_version)"
        else
            print_error "Update failed!"
            return 1
        fi
    else
        print_warning "OpenCode (official) not installed. Install it first."
    fi
}

update_npm() {
    if check_npm_installed; then
        print_status "Updating OpenCode (npm)..."
        npm update -g @opencode/cli
        if [ $? -eq 0 ]; then
            print_status "OpenCode (npm) updated successfully!"
            print_status "New version: $(get_npm_version)"
        else
            print_error "Update failed!"
            return 1
        fi
    else
        print_warning "OpenCode (npm) not installed. Install it first."
    fi
}

update_desktop() {
    if check_desktop_installed; then
        print_status "Updating OpenCode Desktop..."
        # Re-download and reinstall
        install_desktop
    else
        print_warning "OpenCode Desktop not installed. Install it first."
    fi
}

update_all() {
    update_official
    update_npm
    update_desktop
}

show_menu() {
    print_header

    echo -e "${CYAN}Installed Components:${NC}"
    if check_official_installed; then
        echo -e "  Official (CLI): ${GREEN}installed${NC} ($(get_official_version))"
    else
        echo -e "  Official (CLI): ${RED}not installed${NC}"
    fi
    if check_npm_installed; then
        echo -e "  npm (@opencode/cli): ${GREEN}installed${NC} ($(get_npm_version))"
    else
        echo -e "  npm (@opencode/cli): ${RED}not installed${NC}"
    fi
    if check_desktop_installed; then
        echo -e "  Desktop App: ${GREEN}installed${NC} ($(get_desktop_version))"
    else
        echo -e "  Desktop App: ${RED}not installed${NC}"
    fi

    echo ""
    echo "What would you like to do?"
    echo "1) Install"
    echo "2) Uninstall"
    echo "3) Update"
    echo "4) Clear cache"
    echo "5) Check status"
    echo "6) Exit"
    echo ""
}

install_submenu() {
    while true; do
        echo ""
        echo "Install:"
        echo "1) Official install script (recommended)"
        echo "2) npm (@opencode/cli)"
        echo "3) Desktop App (.deb)"
        echo "4) All components"
        echo "5) Back to main menu"
        echo ""
        read -p "Please enter your choice (1-5): " choice
        case $choice in
            1) install_official || true ;;
            2) install_npm || true ;;
            3) install_desktop || true ;;
            4) 
                install_official || true
                echo ""
                install_npm || true
                echo ""
                install_desktop || true
                ;;
            5) return ;;
            *) print_error "Invalid choice!" ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
    done
}

uninstall_submenu() {
    while true; do
        echo ""
        echo "Uninstall:"
        echo "1) Official install script"
        echo "2) npm (@opencode/cli)"
        echo "3) Desktop App"
        echo "4) All (with data cleanup)"
        echo "5) Back to main menu"
        echo ""
        read -p "Please enter your choice (1-5): " choice
        case $choice in
            1)
                if check_official_installed; then
                    read -p "Remove official installation? (y/N): " -n 1 -r; echo
                    [[ $REPLY =~ ^[Yy]$ ]] && uninstall_official || true
                else
                    print_warning "Official installation not found."
                fi
                ;;
            2)
                if check_npm_installed; then
                    read -p "Remove npm installation? (y/N): " -n 1 -r; echo
                    [[ $REPLY =~ ^[Yy]$ ]] && uninstall_npm || true
                else
                    print_warning "npm installation not found."
                fi
                ;;
            3)
                if check_desktop_installed; then
                    read -p "Remove Desktop App? (y/N): " -n 1 -r; echo
                    [[ $REPLY =~ ^[Yy]$ ]] && uninstall_desktop || true
                else
                    print_warning "Desktop App not found."
                fi
                ;;
            4) uninstall_all_opencode || true ;;
            5) return ;;
            *) print_error "Invalid choice!" ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
    done
}

update_submenu() {
    while true; do
        echo ""
        echo "Update:"
        echo "1) Official install script"
        echo "2) npm (@opencode/cli)"
        echo "3) Desktop App"
        echo "4) All components"
        echo "5) Back to main menu"
        echo ""
        read -p "Please enter your choice (1-5): " choice
        case $choice in
            1) update_official || true ;;
            2) update_npm || true ;;
            3) update_desktop || true ;;
            4) update_all || true ;;
            5) return ;;
            *) print_error "Invalid choice!" ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
    done
}

check_status() {
    print_status "Checking OpenCode status..."
    echo ""
    if check_official_installed; then
        echo -e "  Official (CLI): ${GREEN}installed${NC}"
        echo -e "    Version    : $(get_official_version)"
        echo -e "    Location   : $HOME/.opencode/bin/opencode"
    else
        echo -e "  Official (CLI): ${RED}not installed${NC}"
    fi
    echo ""
    if check_npm_installed; then
        echo -e "  npm (@opencode/cli): ${GREEN}installed${NC}"
        echo -e "    Version    : $(get_npm_version)"
        echo -e "    Location   : $(npm root -g 2>/dev/null)/@opencode/cli"
    else
        echo -e "  npm (@opencode/cli): ${RED}not installed${NC}"
    fi
    echo ""
    if check_desktop_installed; then
        echo -e "  Desktop App: ${GREEN}installed${NC}"
        echo -e "    Version    : $(get_desktop_version)"
        echo -e "    Location   : /opt/opencode-desktop.AppImage (or .deb)"
    else
        echo -e "  Desktop App: ${RED}not installed${NC}"
    fi
    echo ""
    if check_opencode_installed; then
        echo -e "  Active in PATH: ${GREEN}yes${NC}"
        echo -e "    Version    : $(get_opencode_version)"
        echo -e "    Location   : $(get_opencode_location)"
    else
        echo -e "  Active in PATH: ${RED}no${NC}"
    fi
}

main() {
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root is not recommended."
        read -p "Continue anyway? (y/N): " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then exit 1; fi
    fi

    while true; do
        show_menu
        read -p "Please enter your choice (1-6): " choice
        case $choice in
            1) install_submenu ;;
            2) uninstall_submenu ;;
            3) update_submenu ;;
            4) clear_cache_only ;;
            5) check_status ;;
            6) print_status "Goodbye!"; exit 0 ;;
            *) print_error "Invalid choice! Please enter 1-6." ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
        echo ""
    done
}

main