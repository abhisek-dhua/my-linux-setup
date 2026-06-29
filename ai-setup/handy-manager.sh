#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

HANDY_DEB_URL="https://github.com/cjpais/Handy/releases/download/v0.8.3/Handy_0.8.3_amd64.deb"
HANDY_DEB_FILE="/tmp/handy_amd64.deb"
HANDY_BIN="/usr/bin/handy"
HANDY_DATA_DIR="$HOME/.local/share/com.pais.handy"

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}    HANDY MANAGER${NC}"
    echo -e "${BLUE}================================${NC}"
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

check_handy_installed() {
    command -v handy &> /dev/null && [ -x "$HANDY_BIN" ]
}

get_handy_version() {
    if check_handy_installed; then
        dpkg -s handy 2>/dev/null | grep -i "^Version:" | cut -d' ' -f2 || echo "unknown"
    else
        echo "not installed"
    fi
}

install_handy() {
    print_status "Installing Handy..."

    if check_handy_installed; then
        print_warning "Handy is already installed. Version: $(get_handy_version)"
        read -p "Do you want to reinstall? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            return
        fi
    fi

    print_status "Installing dependencies..."
    sudo apt-get install -y libgtk-layer-shell0 libappindicator3-1 libwebkit2gtk-4.1-0 libgtk-3-0

    print_status "Installing Wayland/X11 text input tools..."
    sudo apt-get install -y wtype xdotool wl-clipboard || true

    print_status "Downloading Handy v0.8.3..."
    if ! curl -L -o "$HANDY_DEB_FILE" "$HANDY_DEB_URL"; then
        print_error "Failed to download Handy. Check internet connection."
        return 1
    fi

    print_status "Installing Handy..."
    if ! sudo dpkg -i "$HANDY_DEB_FILE"; then
        print_warning "Fixing missing dependencies..."
        sudo apt-get install -f -y
    fi

    rm -f "$HANDY_DEB_FILE"

    if check_handy_installed; then
        print_status "Handy installed successfully!"
        print_status "Version: $(get_handy_version)"
    else
        print_error "Installation failed!"
        return 1
    fi
}

uninstall_handy() {
    if ! check_handy_installed; then
        print_warning "Handy is not installed"
        return
    fi

    print_warning "This will completely remove Handy and its data"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return
    fi

    print_status "Removing Handy package..."
    sudo dpkg -r handy || sudo apt-get remove -y handy

    print_status "Removing Handy data directory..."
    rm -rf "$HANDY_DATA_DIR"

    print_status "Removing systemd service..."
    systemctl --user disable handy.service 2>/dev/null || true
    rm -f "$HOME/.config/systemd/user/handy.service"
    systemctl --user daemon-reload 2>/dev/null || true

    print_status "Removing autostart desktop file..."
    rm -f "$HOME/.config/autostart/Handy.desktop"

    print_status "Removing Wayland env vars from shell config..."
    local SHELL_CONFIG="$HOME/.bashrc"
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    fi
    sed -i '/# Handy Wayland compatibility/,/HANDY_TEXT_INPUT_TOOL=ydotool/d' "$SHELL_CONFIG" 2>/dev/null || true

    print_status "Handy uninstalled successfully"
}

update_handy() {
    if ! check_handy_installed; then
        print_error "Handy is not installed. Please install it first."
        return 1
    fi

    local current_version=$(get_handy_version)
    print_status "Current version: $current_version"
    print_status "Downloading latest version..."

    local latest_deb_url=$(curl -s https://api.github.com/repos/cjpais/Handy/releases/latest | grep "browser_download_url.*amd64\.deb\"" | cut -d'"' -f4)

    if [ -z "$latest_deb_url" ]; then
        print_warning "Could not determine latest version, using v0.8.3"
        latest_deb_url="$HANDY_DEB_URL"
    fi

    print_status "Downloading from: $latest_deb_url"
    if ! curl -L -o "$HANDY_DEB_FILE" "$latest_deb_url"; then
        print_error "Failed to download update."
        return 1
    fi

    print_status "Installing update..."
    if ! sudo dpkg -i "$HANDY_DEB_FILE"; then
        sudo apt-get install -f -y
    fi

    rm -f "$HANDY_DEB_FILE"

    if check_handy_installed; then
        print_status "Handy updated successfully!"
        print_status "New version: $(get_handy_version)"
    else
        print_error "Update failed!"
        return 1
    fi
}

setup_wayland_environment() {
    print_status "Setting up Wayland environment for Handy..."

    local SHELL_CONFIG="$HOME/.bashrc"
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    fi

    if grep -q "WEBKIT_DISABLE_DMABUF_RENDERER" "$SHELL_CONFIG" 2>/dev/null; then
        print_warning "Wayland env vars already set in $SHELL_CONFIG"
    else
        print_status "Adding Wayland compatibility env vars to $SHELL_CONFIG..."
        cat >> "$SHELL_CONFIG" << 'EOF'

# Handy Wayland compatibility
export WEBKIT_DISABLE_DMABUF_RENDERER=1
export HANDY_NO_GTK_LAYER_SHELL=1
export HANDY_TEXT_INPUT_TOOL=ydotool
EOF
        print_status "Environment variables added. Restart your shell or run: source $SHELL_CONFIG"
    fi
}

create_systemd_service() {
    print_status "Creating systemd user service for Handy..."

    local SERVICE_DIR="$HOME/.config/systemd/user"
    local SERVICE_FILE="$SERVICE_DIR/handy.service"

    mkdir -p "$SERVICE_DIR"

    cat > "$SERVICE_FILE" << 'EOF'
[Unit]
Description=Handy Speech-to-Text
After=graphical-session.target
Requires=graphical-session.target

[Service]
Type=simple
Environment=DISPLAY=:0
Environment=HANDY_TEXT_INPUT_TOOL=ydotool
ExecStart=/usr/bin/handy --start-hidden
Restart=on-failure
RestartSec=3

[Install]
WantedBy=graphical-session.target
EOF

    systemctl --user daemon-reload || true
    systemctl --user enable handy.service || true

    print_status "Systemd service created"
    print_status "Enable manually with: systemctl --user enable --now handy"
}

update_autostart_desktop() {
    print_status "Updating Handy autostart for Wayland..."

    local AUTOSTART_DIR="$HOME/.config/autostart"
    local DESKTOP_FILE="$AUTOSTART_DIR/Handy.desktop"

    mkdir -p "$AUTOSTART_DIR"

    cat > "$DESKTOP_FILE" << 'EOF'
[Desktop Entry]
Type=Application
Version=1.0
Name=Handy
Comment=Handy speech-to-text (Wayland compatible)
Exec=env DISPLAY=:0 WEBKIT_DISABLE_DMABUF_RENDERER=1 HANDY_TEXT_INPUT_TOOL=ydotool /usr/bin/handy --start-hidden
StartupNotify=false
Terminal=false
EOF

    print_status "Autostart desktop file updated with Wayland fixes"
    print_status "DISPLAY=:0 will be set automatically on login"
}

show_status() {
    print_header
    echo -e "${BLUE}Current Status:${NC}"
    echo -e "Handy: $(check_handy_installed && echo -e "${GREEN}Installed ($(get_handy_version))${NC}" || echo -e "${RED}Not Installed${NC}")"

    if check_handy_installed; then
        echo
        echo -e "${BLUE}Installation Details:${NC}"
        echo -e "  Binary: ${GREEN}$(which handy)${NC}"
        echo -e "  Data: ${GREEN}$HANDY_DATA_DIR${NC}"
        echo
        echo -e "${BLUE}Process Status:${NC}"
        if pgrep -x handy > /dev/null 2>&1; then
            echo -e "  Status: ${GREEN}Running${NC} (PID: $(pgrep -x handy))"
        else
            echo -e "  Status: ${RED}Not running${NC}"
        fi
    fi
}

show_menu() {
    print_header

    if check_handy_installed; then
        echo -e "${GREEN}Handy is currently installed ($(get_handy_version))${NC}"
    else
        echo -e "${RED}Handy is not installed${NC}"
    fi
    echo
    echo "What would you like to do?"
    echo "1) Install Handy"
    echo "2) Uninstall Handy"
    echo "3) Update Handy"
    echo "4) Show Status"
    echo "5) Apply Fixes (Wayland, systemd, autostart)"
    echo "6) Remove Fixes"
    echo "7) Exit"
    echo
}

apply_fixes() {
    print_status "Installing ydotool..."
    sudo apt-get install -y ydotool || true

    if ! groups | grep -q '\binput\b'; then
        print_status "Adding user to input group for ydotool..."
        sudo usermod -aG input "$USER"
        print_warning "You may need to log out and back in for group changes to take effect"
    fi

    setup_wayland_environment
    create_systemd_service
    update_autostart_desktop
}

remove_fixes() {
    print_status "Removing applied fixes..."

    local SHELL_CONFIG="$HOME/.bashrc"
    if [ -n "$ZSH_VERSION" ] || [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    fi
    sed -i '/# Handy Wayland compatibility/,/HANDY_TEXT_INPUT_TOOL=ydotool/d' "$SHELL_CONFIG" 2>/dev/null || true
    print_status "Removed Wayland env vars from $SHELL_CONFIG"

    systemctl --user disable handy.service 2>/dev/null || true
    rm -f "$HOME/.config/systemd/user/handy.service"
    systemctl --user daemon-reload 2>/dev/null || true
    print_status "Removed systemd service"

    rm -f "$HOME/.config/autostart/Handy.desktop"
    print_status "Removed autostart desktop file"

    print_status "Removing ydotool..."
    sudo apt-get remove -y ydotool 2>/dev/null || true

    print_status "All fixes removed successfully"
}

main() {
    while true; do
        show_menu
        read -p "Please enter your choice (1-7): " choice

        case $choice in
            1) install_handy ;;
            2) uninstall_handy ;;
            3) update_handy ;;
            4) show_status ;;
            5) apply_fixes ;;
            6) remove_fixes ;;
            7)
                print_status "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice! Please enter 1-7."
                ;;
        esac

        echo
        read -p "Press Enter to continue..."
        echo
    done
}

main "$@"
