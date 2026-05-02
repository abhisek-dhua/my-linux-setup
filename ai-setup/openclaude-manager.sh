#!/bin/bash

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

OPENCLAUDE_DIR="$HOME/.openclaude"
CONFIG_FILE="$HOME/.openclaude.json"
PROJECT_DIR="$HOME/.openclaude/projects"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}       OpenClaude Manager v1.0       ${NC}"
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

get_openclaude_path() {
    local name path
    for name in openclaude claude; do
        path=$(command -v "$name" 2>/dev/null) && { echo "$path"; return 0; }
    done

    if command -v npm &>/dev/null; then
        local npm_bin=$(get_npm_bin)
        if [[ -n "$npm_bin" ]]; then
            for name in openclaude claude; do
                [[ -x "$npm_bin/$name" ]] && { echo "$npm_bin/$name"; return 0; }
            done
        fi
        npm list -g --depth=0 @gitlawb/openclaude 2>/dev/null | grep -q '@gitlawb/openclaude' && return 0
    fi
    return 1
}

check_installation() {
    get_openclaude_path >/dev/null
}

confirm() {
    read -p "$1" ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

install_openclaude() {
    section "Install OpenClaude"

    if check_installation; then
        print_warning "OpenClaude appears to be already installed."
        confirm "Do you want to reinstall? (y/N): " || { print_info "Installation cancelled."; pause; return; }
    fi

    if ! command -v npm &>/dev/null; then
        print_error "npm is not installed. Please install Node.js first."
        pause; return
    fi

    print_info "Installing OpenClaude CLI via npm..."
    if npm install -g @gitlawb/openclaude; then
        print_success "OpenClaude installed successfully via @gitlawb/openclaude!"
    else
        print_error "npm installation failed. Check the output above."
        print_info "Make sure you have npm installed and try: npm install -g @gitlawb/openclaude"
    fi
    pause
}

uninstall_openclaude() {
    section "Uninstall OpenClaude"

    if ! check_installation; then
        print_warning "OpenClaude does not appear to be installed."
    fi

    print_warning "This will uninstall OpenClaude AND delete ALL data in $OPENCLAUDE_DIR"
    confirm "Are you sure? (y/N): " || { print_info "Uninstallation cancelled."; pause; return; }

    print_info "Uninstalling OpenClaude via npm..."
    npm uninstall -g @gitlawb/openclaude 2>/dev/null

    rm -f "$HOME/.local/bin/openclaude" "$HOME/.local/bin/claude" 2>/dev/null

    if [[ -d "$OPENCLAUDE_DIR" ]]; then
        rm -rf "$OPENCLAUDE_DIR"
        print_success "Removed entire $OPENCLAUDE_DIR directory"
    fi

    if [[ -f "$CONFIG_FILE" ]]; then
        rm -f "$CONFIG_FILE"
        print_success "Removed $CONFIG_FILE"
    fi

    rm -rf "$HOME/.config/openclaude" "$HOME/.config/claude" 2>/dev/null

    print_success "OpenClaude fully uninstalled - all data cleared!"
    pause
}

clear_all_data() {
    section "Clear All Cache and Config"

    print_warning "This will delete ALL of the following:"
    echo "  - OpenClaude directory ($OPENCLAUDE_DIR)"
    echo "  - Config file ($CONFIG_FILE)"
    echo "  - All sessions, history, cache, memories, projects"
    echo ""

    read -p "Are you absolutely sure? Type 'DELETE' to confirm: " confirm
    if [[ "$confirm" != "DELETE" ]]; then
        print_info "Operation cancelled."
        return
    fi

    print_info "Clearing all OpenClaude data..."

    if [[ -d "$OPENCLAUDE_DIR" ]]; then
        rm -rf "$OPENCLAUDE_DIR"
        print_success "Removed $OPENCLAUDE_DIR"
    else
        print_info "OpenClaude directory not found."
    fi

    if [[ -f "$CONFIG_FILE" ]]; then
        rm -f "$CONFIG_FILE"
        print_success "Removed $CONFIG_FILE"
    fi

    rm -rf "$HOME/.config/openclaude" "$HOME/.config/claude" 2>/dev/null

    print_success "All OpenClaude data cleared!"
    pause
}

clear_cache() {
    section "Clear Cache Only"

    if [[ ! -d "$OPENCLAUDE_DIR" ]]; then
        print_warning "OpenClaude directory not found."
        pause; return
    fi

    print_info "Clearing cache..."

    local items=(
        "sessions:sessions"
        "session-env:session environment"
        "cache:cache"
        "file-history:file history"
        "shell-snapshots:shell snapshots"
        "tasks:tasks"
        "backups:backups"
    )

    local entry path label
    for entry in "${items[@]}"; do
        path="$OPENCLAUDE_DIR/${entry%%:*}"
        label="${entry#*:}"
        rm -rf "$path"/* 2>/dev/null && print_success "Cleared $label" || print_info "No $label to clear"
    done

    rm -f "$OPENCLAUDE_DIR/history.jsonl" 2>/dev/null && print_success "Cleared history" || print_info "No history to clear"

    print_success "Cache cleared!"
    pause
}

clear_projects() {
    section "Clear Project Data"

    if [[ ! -d "$PROJECT_DIR" ]]; then
        print_warning "Project directory not found."
        pause; return
    fi

    print_warning "This will remove all project data including memories."
    confirm "Are you sure? (y/N): " || { print_info "Operation cancelled."; return; }

    rm -rf "$PROJECT_DIR"/*
    print_success "All project data cleared!"
    pause
}

update_openclaude() {
    section "Update OpenClaude"

    if ! command -v npm &>/dev/null; then
        print_error "npm is not installed. Please install Node.js first."
        pause; return
    fi

    print_info "Updating OpenClaude via npm..."
    if npm install -g @gitlawb/openclaude@latest; then
        print_success "OpenClaude updated to latest version via @gitlawb/openclaude!"
    else
        print_error "npm update failed. Check the output above."
    fi
    pause
}

show_status() {
    section "OpenClaude Status"

    local bin_path=$(get_openclaude_path)

    if [[ -n "$bin_path" && -x "$bin_path" ]]; then
        print_success "OpenClaude is installed"
        echo "  Path: $bin_path"
        echo "  Version: $($bin_path --version 2>/dev/null || echo 'unknown')"
    elif check_installation; then
        print_success "OpenClaude is installed"
        echo "  Binary not found in PATH, but npm package is installed."
        echo "  Tip: Add npm global bin to your PATH:"
        echo "       export PATH=\"\$(npm prefix -g)/bin:\$PATH\""
    else
        print_error "OpenClaude is not installed"
    fi

    echo ""
    echo "Data Directory: $OPENCLAUDE_DIR"
    if [[ -d "$OPENCLAUDE_DIR" ]]; then
        echo "  Size: $(du -sh "$OPENCLAUDE_DIR" 2>/dev/null | cut -f1)"
        echo "  Contents:"
        ls -la "$OPENCLAUDE_DIR" 2>/dev/null | tail -n +4 | head -20
    else
        echo "  (not found)"
    fi

    echo ""
    if [[ -f "$CONFIG_FILE" ]]; then
        print_info "Config file exists: $CONFIG_FILE"
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

  1) Install OpenClaude
  2) Uninstall OpenClaude
  3) Clear All Cache and Config
  4) Clear Cache Only
  5) Clear Project Data Only
  6) Update OpenClaude
  7) Show Status
  8) Exit

EOF
        read -p "Enter your choice [1-8]: " choice

        case $choice in
            1) install_openclaude ;;
            2) uninstall_openclaude ;;
            3) clear_all_data ;;
            4) clear_cache ;;
            5) clear_projects ;;
            6) update_openclaude ;;
            7) show_status ;;
            8) print_header; print_success "Goodbye!"; exit 0 ;;
            *) print_error "Invalid option. Please try again."; sleep 1 ;;
        esac
    done
}

main_menu
