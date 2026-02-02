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
    echo -e "${BLUE}  Cursor AI IDE Manager${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to display menu
show_menu() {
    echo -e "${BLUE}What would you like to do?${NC}"
    echo "  1) Install Cursor AI IDE"
    echo "  2) Update Cursor AI IDE"
    echo "  3) Uninstall Cursor AI IDE"
    echo "  4) Exit"
    echo
}

# Function to check if Cursor is installed via apt
is_cursor_installed() {
    dpkg -l | grep -q "^ii.*cursor"
}

# Function to check if Cursor is installed as AppImage
is_appimage_installed() {
    [ -f "/opt/cursor.appimage" ]
}

# Function to check installation method
get_installation_method() {
    if is_cursor_installed; then
        echo "apt"
    elif is_appimage_installed; then
        echo "appimage"
    else
        echo "none"
    fi
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

# Function to install via apt repository
install_via_apt() {
    print_status "Installing Cursor via official apt repository..."
    
    # Add Cursor's GPG key
    print_status "Adding Cursor's GPG key..."
    curl -fsSL https://downloads.cursor.com/keys/anysphere.asc | gpg --dearmor | sudo tee /etc/apt/keyrings/cursor.gpg > /dev/null
    
    # Add the Cursor repository
    print_status "Adding Cursor repository..."
    echo "deb [arch=amd64,arm64 signed-by=/etc/apt/keyrings/cursor.gpg] https://downloads.cursor.com/aptrepo stable main" | sudo tee /etc/apt/sources.list.d/cursor.list > /dev/null
    
    # Update and install
    print_status "Updating package list and installing Cursor..."
    sudo apt update
    sudo apt install -y cursor
    
    print_status "Cursor installed successfully via apt!"
    echo -e "${GREEN}You can now run Cursor by:${NC}"
    echo "  • Searching for 'Cursor' in your applications menu"
    echo "  • Running 'cursor' from terminal"
}

# Function to install via deb file
install_via_deb() {
    print_status "Installing Cursor via .deb file..."
    
    # Check for deb file
    if ! ls *.deb 1> /dev/null 2>&1; then
        print_error "No .deb file found in current directory!"
        echo "Please download a .deb file to the current directory first."
        echo "Expected files: *.deb"
        return 1
    fi
    
    local deb_file=$(ls *.deb | head -n 1)
    print_status "Found .deb file: $deb_file"
    
    print_status "Installing $deb_file..."
    sudo apt install -y "./$deb_file"
    
    print_status "Cursor installed successfully via deb file!"
    echo -e "${GREEN}You can now run Cursor by:${NC}"
    echo "  • Searching for 'Cursor' in your applications menu"
    echo "  • Running 'cursor' from terminal"
}

# Function to install via AppImage
install_via_appimage() {
    print_status "Installing Cursor via AppImage..."
    
    # Check for AppImage file
    if ! find_appimage; then
        print_error "No AppImage file found in current directory!"
        echo "Please download an AppImage file to the current directory first."
        echo "Expected files: *.AppImage or cursor.appimage"
        return 1
    fi
    
    # Update package list and install dependencies
    print_status "Installing dependencies..."
    sudo apt update
    sudo apt install -y curl libfuse2
    
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
        return 0
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
    
    print_status "Cursor installed successfully via AppImage!"
    echo -e "${GREEN}You can now run Cursor by:${NC}"
    echo "  • Searching for 'Cursor' in your applications menu"
    echo "  • Running '/opt/cursor.appimage' from terminal"
    echo "  • Double-clicking the desktop launcher"
}

# Function to install cursor
install_cursor() {
    print_header
    print_status "Starting installation process..."
    
    # Check if already installed
    local method=$(get_installation_method)
    if [ "$method" != "none" ]; then
        print_warning "Cursor is already installed via $method method!"
        echo "Would you like to:"
        echo "  1) Update instead"
        echo "  2) Continue with fresh installation (not recommended)"
        echo "  3) Cancel"
        read -p "Choose an option (1/2/3): " -n 1 -r
        echo
        case $REPLY in
            1)
                update_cursor
                return
                ;;
            3)
                print_status "Installation cancelled."
                return
                ;;
            2)
                print_warning "Proceeding with fresh installation..."
                ;;
            *)
                print_status "Installation cancelled."
                return
                ;;
        esac
    fi
    
    echo -e "${BLUE}Choose installation method:${NC}"
    echo "  1) Install via official apt repository (recommended)"
    echo "  2) Install via .deb file (if downloaded)"
    echo "  3) Install via AppImage (portable)"
    echo "  4) Cancel"
    echo
    read -p "Choose an option (1/2/3/4): " -n 1 -r
    echo
    
    case $REPLY in
        1)
            install_via_apt
            ;;
        2)
            install_via_deb
            ;;
        3)
            install_via_appimage
            ;;
        4)
            print_status "Installation cancelled."
            ;;
        *)
            print_error "Invalid option."
            ;;
    esac
}

# Function to update cursor
update_cursor() {
    print_header
    print_status "Starting update process..."
    
    local method=$(get_installation_method)
    
    case $method in
        "apt")
            print_status "Updating Cursor via apt repository..."
            sudo apt update
            sudo apt install --only-upgrade cursor
            print_status "Cursor updated successfully!"
            ;;
        "appimage")
            print_status "Updating Cursor via AppImage..."
            print_warning "AppImage updates require manual process."
            echo "Options:"
            echo "  1) Download new AppImage and reinstall (recommended)"
            echo "  2) Cancel"
            read -p "Choose an option (1/2): " -n 1 -r
            echo
            case $REPLY in
                1)
                    install_via_appimage
                    ;;
                2)
                    print_status "Update cancelled."
                    ;;
                *)
                    print_error "Invalid option."
                    ;;
            esac
            ;;
        "none")
            print_error "Cursor is not installed!"
            echo "Would you like to install it instead?"
            read -p "Install now? (y/n): " -n 1 -r
            echo
            case $REPLY in
                [Yy])
                    install_cursor
                    ;;
                *)
                    print_status "Operation cancelled."
                    ;;
            esac
            ;;
    esac
}

# Function to uninstall cursor
uninstall_cursor() {
    print_header
    print_status "Starting uninstallation process..."
    
    local method=$(get_installation_method)
    
    if [ "$method" = "none" ]; then
        print_error "Cursor is not installed!"
        return
    fi
    
    print_warning "This will remove Cursor from your system."
    read -p "Are you sure you want to continue? (y/n): " -n 1 -r
    echo
    
    case $REPLY in
        [Yy])
            ;;
        *)
            print_status "Uninstallation cancelled."
            return
            ;;
    esac
    
    case $method in
        "apt")
            print_status "Removing Cursor via apt..."
            sudo apt remove cursor
            sudo apt autoremove
            # Remove repository
            sudo rm -f /etc/apt/sources.list.d/cursor.list
            sudo rm -f /etc/apt/keyrings/cursor.gpg
            print_status "Cursor uninstalled successfully!"
            ;;
        "appimage")
            print_status "Removing Cursor AppImage..."
            # Kill running processes
            if lsof /opt/cursor.appimage 1>/dev/null 2>&1; then
                print_status "Stopping running Cursor processes..."
                sudo fuser -k /opt/cursor.appimage
            fi
            sudo rm -f /opt/cursor.appimage
            sudo rm -f /opt/cursor.png
            sudo rm -f /usr/share/applications/cursor.desktop
            sudo update-desktop-database /usr/share/applications
            print_status "Cursor AppImage uninstalled successfully!"
            ;;
    esac
}

# Main function
main() {
    print_header
    
    # Check if not running as root
    check_root
    
    # Show menu
    while true; do
        show_menu
        read -p "Enter your choice (1-4): " -n 1 -r
        echo
        
        case $REPLY in
            1)
                install_cursor
                return
                ;;
            2)
                update_cursor
                return
                ;;
            3)
                uninstall_cursor
                return
                ;;
            4)
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid option. Please choose 1-4."
                echo
                ;;
        esac
    done
}

# Run main function
main "$@"

