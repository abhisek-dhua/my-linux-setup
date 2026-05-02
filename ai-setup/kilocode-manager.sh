#!/bin/bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

KILO_DIR="$HOME/.config/kilo"
KILOCODE_DIR="$HOME/.kilocode"
KILO_DIR_ALT="$HOME/.kilo"
CONFIG_FILE="$HOME/.config/kilo/kilo.json"
CONFIG_FILEC="$HOME/.config/kilo/kilo.jsonc"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}       KiloCode Manager v1.0        ${NC}"
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

get_kilocode_path() {
    local name path
    for name in kilo kilocode; do
        path=$(command -v "$name" 2>/dev/null) && { echo "$path"; return 0; }
    done

    if command -v npm &>/dev/null; then
        local npm_bin=$(get_npm_bin)
        if [[ -n "$npm_bin" ]]; then
            for name in kilo kilocode; do
                [[ -x "$npm_bin/$name" ]] && { echo "$npm_bin/$name"; return 0; }
            done
        fi
        npm list -g --depth=0 @kilocode/cli 2>/dev/null | grep -q '@kilocode/cli' && return 0
    fi
    return 1
}

check_installation() {
    get_kilocode_path >/dev/null
}

confirm() {
    read -p "$1" ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

install_kilocode() {
    section "Install KiloCode"

    if check_installation; then
        print_warning "KiloCode appears to be already installed."
        confirm "Do you want to reinstall? (y/N): " || { print_info "Installation cancelled."; pause; return; }
    fi

    if ! command -v npm &>/dev/null; then
        print_error "npm is not installed. Please install Node.js first."
        pause; return
    fi

    print_info "Installing KiloCode CLI via npm..."
    if npm install -g @kilocode/cli; then
        print_success "KiloCode installed successfully via @kilocode/cli!"
    else
        print_error "npm installation failed. Check the output above."
        print_info "Make sure you have npm installed and try: npm install -g @kilocode/cli"
    fi
    pause
}

uninstall_kilocode() {
    section "Uninstall KiloCode"

    if ! check_installation; then
        print_warning "KiloCode does not appear to be installed."
    fi

    print_warning "This will uninstall KiloCode AND delete ALL data in config directories"
    confirm "Are you sure? (y/N): " || { print_info "Uninstallation cancelled."; pause; return; }

    print_info "Uninstalling KiloCode via npm..."
    npm uninstall -g @kilocode/cli 2>/dev/null

    rm -f "$HOME/.local/bin/kilo" "$HOME/.local/bin/kilocode" 2>/dev/null

    for dir in "$KILO_DIR" "$KILOCODE_DIR" "$KILO_DIR_ALT"; do
        if [[ -d "$dir" ]]; then
            rm -rf "$dir"
            print_success "Removed $dir"
        fi
    done

    rm -f "$CONFIG_FILE" "$CONFIG_FILEC" 2>/dev/null

    print_success "KiloCode fully uninstalled - all data cleared!"
    pause
}

clear_all_data() {
    section "Clear All Cache and Config"

    print_warning "This will delete ALL of the following:"
    echo "  - KiloCode config ($KILO_DIR)"
    echo "  - KiloCode data ($KILOCODE_DIR, $KILO_DIR_ALT)"
    echo "  - All sessions, history, cache, memories, settings"
    echo ""

    read -p "Are you absolutely sure? Type 'DELETE' to confirm: " confirm
    if [[ "$confirm" != "DELETE" ]]; then
        print_info "Operation cancelled."
        return
    fi

    print_info "Clearing all KiloCode data..."

    for dir in "$KILO_DIR" "$KILOCODE_DIR" "$KILO_DIR_ALT"; do
        if [[ -d "$dir" ]]; then
            rm -rf "$dir"
            print_success "Removed $dir"
        else
            print_info "$dir not found."
        fi
    done

    print_success "All KiloCode data cleared!"
    pause
}

clear_cache() {
    section "Clear Cache Only"

    local found=0
    for dir in "$KILO_DIR" "$KILOCODE_DIR" "$KILO_DIR_ALT"; do
        [[ -d "$dir" ]] && { found=1; break; }
    done

    if [[ $found -eq 0 ]]; then
        print_warning "KiloCode config directories not found."
        pause; return
    fi

    print_info "Clearing cache..."

    local cache_dirs=("cache" "sessions" "session-env" "file-history" "shell-snapshots" "tasks" "backups")
    local dir label

    for dir in "$KILO_DIR" "$KILOCODE_DIR" "$KILO_DIR_ALT"; do
        [[ -d "$dir" ]] || continue
        for label in "${cache_dirs[@]}"; do
            rm -rf "$dir/$label"/* 2>/dev/null && print_success "Cleared $label in $dir" || true
        done
        rm -f "$dir/history.jsonl" 2>/dev/null && print_success "Cleared history in $dir" || true
    done

    print_success "Cache cleared!"
    pause
}

clear_projects() {
    section "Clear Project Data"

    local project_dirs=(".kilo" ".kilocode" ".opencode")
    local found=0

    for dir in "${project_dirs[@]}"; do
        [[ -d "$dir" ]] && { found=1; break; }
    done

    if [[ $found -eq 0 ]]; then
        print_warning "No project data directories found."
        pause; return
    fi

    print_warning "This will remove all project data including memories from current directory."
    confirm "Are you sure? (y/N): " || { print_info "Operation cancelled."; return; }

    for dir in "${project_dirs[@]}"; do
        [[ -d "$dir" ]] && rm -rf "$dir" && print_success "Removed ./$dir"
    done

    print_success "All project data cleared!"
    pause
}

update_kilocode() {
    section "Update KiloCode"

    if ! command -v npm &>/dev/null; then
        print_error "npm is not installed. Please install Node.js first."
        pause; return
    fi

    print_info "Updating KiloCode via npm..."
    if npm install -g @kilocode/cli@latest; then
        print_success "KiloCode updated to latest version via @kilocode/cli!"
    else
        print_error "npm update failed. Check the output above."
    fi
    pause
}

show_status() {
    section "KiloCode Status"

    local bin_path=$(get_kilocode_path)

    if [[ -n "$bin_path" && -x "$bin_path" ]]; then
        print_success "KiloCode is installed"
        echo "  Path: $bin_path"
        echo "  Version: $($bin_path --version 2>/dev/null || echo 'unknown')"
    elif check_installation; then
        print_success "KiloCode is installed"
        echo "  Binary not found in PATH, but npm package is installed."
        echo "  Tip: Add npm global bin to your PATH:"
        echo "       export PATH=\"\$(npm prefix -g)/bin:\$PATH\""
    else
        print_error "KiloCode is not installed"
    fi

    echo ""
    echo "Config Directory: $KILO_DIR"
    if [[ -d "$KILO_DIR" ]]; then
        echo "  Size: $(du -sh "$KILO_DIR" 2>/dev/null | cut -f1)"
        echo "  Contents:"
        ls -la "$KILO_DIR" 2>/dev/null | tail -n +4 | head -20
    else
        echo "  (not found)"
    fi

    for dir in "$KILOCODE_DIR" "$KILO_DIR_ALT"; do
        if [[ -d "$dir" ]]; then
            echo ""
            echo "Additional Directory: $dir"
            echo "  Size: $(du -sh "$dir" 2>/dev/null | cut -f1)"
        fi
    done

    echo ""
    if [[ -f "$CONFIG_FILE" || -f "$CONFIG_FILEC" ]]; then
        print_info "Config file exists"
    else
        print_info "No config file found"
    fi

    pause
}

main_menu() {
    while true; do
        print_header
        cat <<EOF
Please select an option:

  1) Install KiloCode
  2) Uninstall KiloCode
  3) Clear All Cache and Config
  4) Clear Cache Only
  5) Clear Project Data Only
  6) Update KiloCode
  7) Show Status
  8) Exit

EOF
        read -p "Enter your choice [1-8]: " choice

        case $choice in
            1) install_kilocode ;;
            2) uninstall_kilocode ;;
            3) clear_all_data ;;
            4) clear_cache ;;
            5) clear_projects ;;
            6) update_kilocode ;;
            7) show_status ;;
            8) print_header; print_success "Goodbye!"; exit 0 ;;
            *) print_error "Invalid option. Please try again."; sleep 1 ;;
        esac
    done
}

main_menu
