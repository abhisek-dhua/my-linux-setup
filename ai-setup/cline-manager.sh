#!/bin/bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

CLINE_DIR="${CLINE_DIR:-$HOME/.cline}"
CONFIG_DIR="$CLINE_DIR/data"
LOG_DIR="$CLINE_DIR/log"
SETTINGS_FILE="$CONFIG_DIR/settings/cline_mcp_settings.json"
GLOBAL_STATE="$CONFIG_DIR/globalState.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}        Cline Manager v1.0          ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ $1${NC}"; }
pause()         { read -p "Press Enter to continue..."; }

section() {
    print_header
    echo -e "${YELLOW}--- $1 ---${NC}"
    echo ""
}

get_npm_bin() {
    npm bin -g 2>/dev/null || echo "$(npm prefix -g 2>/dev/null)/bin"
}

get_cline_path() {
    local path
    path=$(command -v cline 2>/dev/null) && { echo "$path"; return 0; }

    if command -v npm &>/dev/null; then
        local npm_bin=$(get_npm_bin)
        if [[ -n "$npm_bin" && -x "$npm_bin/cline" ]]; then
            echo "$npm_bin/cline"
            return 0
        fi
        npm list -g --depth=0 cline 2>/dev/null | grep -q 'cline@' && return 0
    fi
    return 1
}

check_installation() {
    get_cline_path >/dev/null
}

confirm() {
    read -p "$1" ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

install_cline() {
    section "Install Cline"

    if check_installation; then
        print_warning "Cline appears to be already installed."
        confirm "Do you want to reinstall? (y/N): " || { print_info "Installation cancelled."; pause; return; }
    fi

    if ! command -v npm &>/dev/null; then
        print_error "npm is not installed. Please install Node.js first."
        pause; return
    fi

    print_info "Installing Cline CLI via npm..."
    if npm install -g cline; then
        print_success "Cline installed successfully!"
    else
        print_error "npm installation failed. Check the output above."
        print_info "Make sure you have npm installed and try: npm install -g cline"
    fi
    pause
}

uninstall_cline() {
    section "Uninstall Cline"

    if ! check_installation; then
        print_warning "Cline does not appear to be installed."
    fi

    print_warning "This will uninstall Cline AND delete ALL data in $CLINE_DIR"
    confirm "Are you sure? (y/N): " || { print_info "Uninstallation cancelled."; pause; return; }

    print_info "Uninstalling Cline via npm..."
    npm uninstall -g cline 2>/dev/null

    rm -f "$HOME/.local/bin/cline" 2>/dev/null

    if [[ -d "$CLINE_DIR" ]]; then
        rm -rf "$CLINE_DIR"
        print_success "Removed entire $CLINE_DIR directory"
    fi

    print_success "Cline fully uninstalled - all data cleared!"
    pause
}

clear_all_data() {
    section "Clear All Cache and Config"

    print_warning "This will delete ALL of the following:"
    echo "  - Cline directory ($CLINE_DIR)"
    echo "  - All config, cache, tasks, logs, secrets"
    echo ""

    read -p "Are you absolutely sure? Type 'DELETE' to confirm: " confirm
    if [[ "$confirm" != "DELETE" ]]; then
        print_info "Operation cancelled."
        return
    fi

    print_info "Clearing all Cline data..."

    if [[ -d "$CLINE_DIR" ]]; then
        rm -rf "$CLINE_DIR"
        print_success "Removed $CLINE_DIR"
    else
        print_info "Cline directory not found."
    fi

    print_success "All Cline data cleared!"
    pause
}

clear_cache() {
    section "Clear Cache Only"

    if [[ ! -d "$CONFIG_DIR" ]]; then
        print_warning "Cline config directory not found."
        pause; return
    fi

    print_info "Clearing cache..."

    local cache_dirs=("cache" "tasks" "workspace" "sessions" "session-env")
    local dir label

    for label in "${cache_dirs[@]}"; do
        rm -rf "$CONFIG_DIR/$label"/* 2>/dev/null && print_success "Cleared $label" || print_info "No $label to clear"
    done

    rm -f "$CONFIG_DIR/history.jsonl" "$CONFIG_DIR/.history" 2>/dev/null && print_success "Cleared history" || true

    if [[ -d "$LOG_DIR" ]]; then
        rm -rf "$LOG_DIR"/*
        print_success "Cleared logs"
    fi

    print_success "Cache cleared!"
    pause
}

clear_projects() {
    section "Clear Project Data"

    local project_configs=(".cline" "cline.json" ".config/cline")
    local found=0

    for dir in "${project_configs[@]}"; do
        [[ -d "$dir" || -f "$dir" ]] && { found=1; break; }
    done

    if [[ $found -eq 0 ]]; then
        print_warning "No project config found in current directory."
        pause; return
    fi

    print_warning "This will remove project-level Cline data from current directory."
    confirm "Are you sure? (y/N): " || { print_info "Operation cancelled."; return; }

    for dir in "${project_configs[@]}"; do
        [[ -d "$dir" ]] && rm -rf "$dir" && print_success "Removed ./$dir"
        [[ -f "$dir" ]] && rm -f "$dir" && print_success "Removed ./$dir"
    done

    print_success "All project data cleared!"
    pause
}

update_cline() {
    section "Update Cline"

    if ! command -v npm &>/dev/null; then
        print_error "npm is not installed. Please install Node.js first."
        pause; return
    fi

    print_info "Updating Cline via npm..."
    if npm install -g cline@latest; then
        print_success "Cline updated to latest version!"
    else
        print_error "npm update failed. Check the output above."
    fi
    pause
}

show_status() {
    section "Cline Status"

    local bin_path=$(get_cline_path)

    if [[ -n "$bin_path" && -x "$bin_path" ]]; then
        print_success "Cline is installed"
        echo "  Path: $bin_path"
        echo "  Version: $($bin_path --version 2>/dev/null || echo 'unknown')"
    elif check_installation; then
        print_success "Cline is installed"
        echo "  Binary not found in PATH, but npm package is installed."
        echo "  Tip: Add npm global bin to your PATH:"
        echo "       export PATH=\"\$(npm prefix -g)/bin:\$PATH\""
    else
        print_error "Cline is not installed"
    fi

    echo ""
    echo "Config Directory: $CONFIG_DIR"
    if [[ -d "$CONFIG_DIR" ]]; then
        echo "  Size: $(du -sh "$CONFIG_DIR" 2>/dev/null | cut -f1)"
        echo "  Contents:"
        ls -la "$CONFIG_DIR" 2>/dev/null | tail -n +4 | head -20
    else
        echo "  (not found)"
    fi

    echo ""
    if [[ -f "$SETTINGS_FILE" ]]; then
        print_info "MCP Settings: $SETTINGS_FILE"
    fi
    if [[ -f "$GLOBAL_STATE" ]]; then
        print_info "Global State: $GLOBAL_STATE"
    fi

    echo ""
    if [[ -d "$LOG_DIR" ]]; then
        echo "Log Directory: $LOG_DIR"
        echo "  Size: $(du -sh "$LOG_DIR" 2>/dev/null | cut -f1)"
    fi

    pause
}

main_menu() {
    while true; do
        print_header
        cat <<EOF
Please select an option:

  1) Install Cline
  2) Uninstall Cline
  3) Clear All Cache and Config
  4) Clear Cache Only
  5) Clear Project Data Only
  6) Update Cline
  7) Show Status
  8) Exit

EOF
        read -p "Enter your choice [1-8]: " choice

        case $choice in
            1) install_cline ;;
            2) uninstall_cline ;;
            3) clear_all_data ;;
            4) clear_cache ;;
            5) clear_projects ;;
            6) update_cline ;;
            7) show_status ;;
            8) print_header; print_success "Goodbye!"; exit 0 ;;
            *) print_error "Invalid option. Please try again."; sleep 1 ;;
        esac
    done
}

main_menu
