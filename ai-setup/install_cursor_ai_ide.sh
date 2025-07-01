#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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
    echo -e "${BLUE}  Cursor AI IDE Installer${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root!"
        exit 1
    fi
}

# Function to check if AppImage file exists
find_appimage() {
    local appimage_file=""
    
    # Check for any .AppImage file
    if ls *.AppImage 1> /dev/null 2>&1; then
        appimage_file=$(ls *.AppImage | head -n 1)
        print_status "Found AppImage file: $appimage_file"
        return 0
    elif [ -f "cursor.appimage" ]; then
        appimage_file="cursor.appimage"
        print_status "Found existing cursor.appimage file"
        return 0
    else
        return 1
    fi
}

# Function to backup existing installation
backup_existing() {
    if [ -f "/opt/cursor.appimage" ]; then
        print_warning "Existing Cursor installation found in /opt/"
        
        # Get file sizes for comparison
        local existing_size=$(stat -c%s /opt/cursor.appimage 2>/dev/null || echo "0")
        local new_size=$(stat -c%s "$1" 2>/dev/null || echo "0")
        
        echo "Existing file size: $(numfmt --to=iec $existing_size)"
        echo "New file size: $(numfmt --to=iec $new_size)"
        
        if [ "$existing_size" = "$new_size" ]; then
            print_warning "Files appear to be the same size - this might be the same version"
        fi
        
        echo
        echo "Options:"
        echo "  y - Update Cursor (backup existing first)"
        echo "  n - Skip update (keep existing version)"
        echo "  f - Force update (overwrite without backup)"
        echo "  c - Cancel installation"
        
        while true; do
            read -p "What would you like to do? (y/n/f/c): " -n 1 -r
            echo
            case $REPLY in
                [Yy])
                    sudo cp /opt/cursor.appimage /opt/cursor.appimage.backup.$(date +%Y%m%d_%H%M%S)
                    print_status "Backup created: /opt/cursor.appimage.backup.$(date +%Y%m%d_%H%M%S)"
                    print_status "Proceeding with update..."
                    return 0
                    ;;
                [Nn])
                    print_status "Skipping update - keeping existing version"
                    return 1
                    ;;
                [Ff])
                    print_warning "Force updating - no backup will be created"
                    return 0
                    ;;
                [Cc])
                    print_status "Installation cancelled by user"
                    exit 0
                    ;;
                *)
                    echo "Invalid option. Please choose y, n, f, or c."
                    ;;
            esac
        done
    fi
    return 0
}

# Function to check if /opt/cursor.appimage is in use
check_appimage_in_use() {
    if lsof /opt/cursor.appimage 1>/dev/null 2>&1; then
        print_warning "/opt/cursor.appimage is currently in use!"
        echo
        echo "The following processes are using /opt/cursor.appimage:"
        lsof /opt/cursor.appimage
        echo
        echo "Please close the Cursor application before proceeding."
        echo "Options:"
        echo "  k - Kill all processes using /opt/cursor.appimage"
        echo "  c - Cancel installation"
        while true; do
            read -p "What would you like to do? (k/c): " -n 1 -r
            echo
            case $REPLY in
                [Kk])
                    print_status "Killing all processes using /opt/cursor.appimage..."
                    sudo fuser -k /opt/cursor.appimage
                    print_status "All processes killed. Proceeding with installation."
                    break
                    ;;
                [Cc])
                    print_status "Installation cancelled by user."
                    exit 0
                    ;;
                *)
                    echo "Invalid option. Please choose k or c."
                    ;;
            esac
        done
    fi
}

# Main installation function
main() {
    print_header
    
    # Check if not running as root
    check_root
    
    print_status "Starting installation process..."
    
    # Update package list and install dependencies
    print_status "Installing dependencies..."
    sudo apt update
    sudo apt install -y curl libfuse2
    
    # Check for AppImage file
    print_status "Checking for AppImage file..."
    if ! find_appimage; then
        print_error "No AppImage file found in current directory!"
        echo "Please download an AppImage file to the current directory first."
        echo "Expected files: *.AppImage or cursor.appimage"
        exit 1
    fi
    
    # Get the AppImage file name
    local appimage_file=""
    if ls *.AppImage 1> /dev/null 2>&1; then
        appimage_file=$(ls *.AppImage | head -n 1)
    else
        appimage_file="cursor.appimage"
    fi
    
    # Backup existing installation
    if ! backup_existing "$appimage_file"; then
        print_status "Keeping existing installation. Exiting..."
        exit 0
    fi
    
    # Check if /opt/cursor.appimage is in use before copying
    check_appimage_in_use
    
    # Create a copy of AppImage file and install it
    print_status "Creating copy of AppImage for installation..."
    if [ "$appimage_file" != "cursor.appimage" ]; then
        # Create a copy of the original AppImage file
        cp "$appimage_file" cursor.appimage
        print_status "Created copy: $appimage_file → cursor.appimage"
        # Install the copy to /opt/
        sudo cp cursor.appimage /opt/cursor.appimage
        print_status "Installed copy to /opt/cursor.appimage"
        # Clean up the temporary copy
        rm cursor.appimage
        print_status "Cleaned up temporary copy"
    else
        # If it's already named cursor.appimage, just copy it
        sudo cp cursor.appimage /opt/cursor.appimage
        print_status "Copied cursor.appimage to /opt/cursor.appimage"
    fi
    
    # Set executable permissions
    sudo chmod +x /opt/cursor.appimage
    
    # Download icon
    print_status "Downloading application icon..."
    if sudo curl -L https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/cursor.png -o /opt/cursor.png; then
        print_status "Icon downloaded successfully"
    else
        print_warning "Failed to download icon, continuing without custom icon"
    fi
    
    # Create desktop launcher
    print_status "Creating desktop launcher..."
    sudo tee /usr/share/applications/cursor.desktop <<EOF
[Desktop Entry]
Name=Cursor AI IDE
Comment=AI-powered code editor
Exec=/opt/cursor.appimage --no-sandbox
Icon=/opt/cursor.png
Type=Application
Categories=Development;IDE;
Keywords=code;editor;ai;development;
StartupWMClass=Cursor
EOF
    
    # Update desktop database
    sudo update-desktop-database /usr/share/applications
    
    print_status "Installation completed successfully!"
    echo
    echo -e "${GREEN}You can now run Cursor by:${NC}"
    echo "  • Searching for 'Cursor' in your applications menu"
    echo "  • Running 'cursor' from terminal (if PATH is set)"
    echo "  • Double-clicking the desktop launcher"
    echo
    print_status "Installation directory: /opt/cursor.appimage"
    print_status "Original file preserved: $appimage_file (in current directory)"
}

# Run main function
main "$@"

