#!/bin/bash

# ZCode (GLM) Installer/Uninstaller Script
# Manages the official ZCode desktop app (GLM-5.3 harness) from z.ai

set -e

# Colors for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PACKAGE_NAME="zcode"
DEFAULT_VERSION="3.7.7"
DOWNLOADS_DIR="$HOME/Downloads"

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}   ZCode (GLM) Manager${NC}"
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

get_arch() {
    case "$(uname -m)" in
        x86_64)  echo "x64" ;;
        aarch64) echo "arm64" ;;
        *)
            print_error "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac
}

get_deb_url() {
    local version="${1:-$DEFAULT_VERSION}"
    local arch="$(get_arch)"
    echo "https://cdn-zcode.z.ai/zcode/electron/releases/${version}/linux-${arch}/ZCode-${version}-linux-${arch}.deb"
}

check_zcode_installed() {
    # Method 1: command -v
    if command -v zcode &> /dev/null; then
        # Additional check: confirm the package is installed via dpkg
        if dpkg -s "$PACKAGE_NAME" &> /dev/null; then
            return 0
        fi
    fi

    # Method 2: dpkg status check (covers broken symlink cases)
    if dpkg -s "$PACKAGE_NAME" &> /dev/null; then
        if dpkg -s "$PACKAGE_NAME" | grep -q "Status: install ok installed"; then
            return 0
        fi
    fi

    # Method 3: check the binary path
    if [ -x "/opt/ZCode/zcode" ]; then
        return 0
    fi

    return 1
}

get_installed_version() {
    dpkg -s "$PACKAGE_NAME" 2>/dev/null | grep '^Version:' | awk '{print $2}' || echo "unknown"
}

ensure_sudo() {
    if [ "$EUID" -eq 0 ]; then
        print_warning "Running as root. No sudo needed."
        return 0
    fi
    if ! sudo -n true 2> /dev/null; then
        print_status "Sudo required for installation."
        sudo true
    fi
}

install_zcode() {
    print_status "Installing ZCode..."
    print_status "Architecture: $(get_arch)"

    # Check for required tools
    if ! command -v curl &> /dev/null; then
        print_error "curl is not installed. Please install curl first."
        exit 1
    fi

    local version="${ZCODE_VERSION:-$DEFAULT_VERSION}"
    local url="$(get_deb_url "$version")"
    local deb_file="$DOWNLOADS_DIR/ZCode-${version}-linux-$(get_arch).deb"

    # Download (or reuse existing backup)
    if [ -f "$deb_file" ]; then
        print_status "Using existing .deb backup: $deb_file"
    else
        print_status "Downloading ZCode v${version} from $url ..."
        mkdir -p "$DOWNLOADS_DIR"
        curl -fL -o "$deb_file" "$url"
        print_status "Downloaded and backed up to: $deb_file"
    fi

    ensure_sudo
    print_status "Installing $deb_file ..."
    sudo dpkg -i "$deb_file"

    # Fix any missing dependencies
    if ! dpkg -s "$PACKAGE_NAME" &> /dev/null; then
        print_status "Fixing dependencies..."
        sudo apt-get -f install -y
    fi

    if check_zcode_installed; then
        print_status "ZCode installed successfully! Version: $(get_installed_version)"
        echo -e "${GREEN}You can now run 'zcode' from your terminal or launch it from the app menu.${NC}"
    else
        print_error "Installation failed!"
        exit 1
    fi
}

uninstall_zcode() {
    print_status "Uninstalling ZCode..."

    if ! check_zcode_installed; then
        print_warning "ZCode is not installed!"
        return
    fi

    read -p "This will remove ZCode and all its data. Continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Uninstallation cancelled."
        return
    fi

    # Remove the package
    if dpkg -s "$PACKAGE_NAME" &> /dev/null; then
        ensure_sudo
        print_status "Removing package..."
        sudo dpkg -r "$PACKAGE_NAME"
        sudo apt-get autoremove -y 2>/dev/null || true
    fi

    # Remove local data and config directories
    print_status "Removing ZCode data directories..."

    local dirs=(
        "$HOME/.zcode"
        "$HOME/.config/ZCode"
        "$HOME/.config/zcode"
        "$HOME/.local/share/ZCode"
        "$HOME/.cache/ZCode"
    )

    for dir in "${dirs[@]}"; do
        if [ -d "$dir" ]; then
            print_status "Removing: $dir"
            rm -rf "$dir"
        else
            print_warning "Directory not found: $dir"
        fi
    done

    read -p "Do you also want to delete the downloaded .deb backup in $DOWNLOADS_DIR? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -f "$DOWNLOADS_DIR"/ZCode-*.deb
        print_status "Removed .deb backups."
    fi

    print_status "ZCode uninstallation complete!"
}

clear_cache_only() {
    print_status "Clearing ZCode cache and configuration..."

    # Remove cache and config directories only (keeping the application)
    local dirs=(
        "$HOME/.config/ZCode"
        "$HOME/.config/zcode"
        "$HOME/.local/share/ZCode"
        "$HOME/.cache/ZCode"
        "$HOME/.zcode/v2"
        "$HOME/.zcode/workspace"
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
    print_warning "You will need to log in again on next launch."
}

update_zcode() {
    print_status "Updating ZCode..."

    if ! check_zcode_installed; then
        print_error "ZCode is not installed. Please install it first."
        return 1
    fi

    local current="$(get_installed_version)"
    local version="${ZCODE_VERSION:-$DEFAULT_VERSION}"

    print_status "Current version: $current"
    print_status "Latest known version: $version"

    if [[ "$current" == "$version"* ]]; then
        print_status "You already have the latest known version ($version)."
        read -p "Force reinstall anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Update cancelled."
            return
        fi
    fi

    install_zcode
    print_status "ZCode updated successfully! New version: $(get_installed_version)"
}

show_menu() {
    print_header

    if check_zcode_installed; then
        echo -e "${GREEN}ZCode is currently installed${NC} (version: $(get_installed_version))"
    else
        echo -e "${RED}ZCode is not installed${NC}"
    fi
    echo ""
    echo "What would you like to do?"
    echo "1) Install ZCode"
    echo "2) Uninstall ZCode"
    echo "3) Clear cache and config only"
    echo "4) Update ZCode"
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
                if check_zcode_installed; then
                    print_warning "ZCode is already installed!"
                    read -p "Do you want to reinstall it? (y/N): " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        uninstall_zcode
                        install_zcode
                    fi
                else
                    install_zcode
                fi
                ;;
            2)
                if check_zcode_installed; then
                    uninstall_zcode
                else
                    print_warning "ZCode is not installed!"
                fi
                ;;
            3)
                clear_cache_only
                ;;
            4)
                update_zcode
                ;;
            5)
                if check_zcode_installed; then
                    print_status "ZCode is installed"
                    print_status "Version: $(get_installed_version)"
                    print_status "Binary: $(command -v zcode || echo '/opt/ZCode/zcode')"
                    print_status "Data directory: $HOME/.zcode"
                else
                    print_warning "ZCode is not installed"
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