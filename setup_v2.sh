#!/bin/bash

################################################################################
# Linux Setup Script v2.0 - TUI Installation Tool
# Based on updated analysis with modular, selectable categories
#
# Features:
# - TUI interface with whiptail
# - Category-based installation selection
# - Modular installation functions
# - Progress tracking and logging
# - Skip already-installed items
# - Error handling with user choice to continue
#
# Run with: sudo ./setup_v2.sh
################################################################################

#==============================================================================
# CONFIGURATION & GLOBALS
#==============================================================================

# Version and script info
SCRIPT_VERSION="2.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="/tmp/setup_v2_$(date +%Y%m%d_%H%M%S).log"

# Target user (will be prompted)
TARGET_USER=""
TARGET_HOME=""

# Whiptail settings  
HEIGHT=24
WIDTH=80
MENU_HEIGHT=15

# Status tracking
EXIT_CODE=0
COMPLETED_ITEMS=()
SKIPPED_ITEMS=()
FAILED_ITEMS=()

#==============================================================================
# UTILITY FUNCTIONS
#==============================================================================

# Logging function
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "${LOG_FILE}"
}

info() { log "INFO" "$*"; }
warn() { log "WARN" "$*"; }
error() { log "ERROR" "$*"; }
debug() { log "DEBUG" "$*"; }

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        whiptail --title "Error" --msgbox "\nThis script must be run with sudo:\n\n  sudo ./setup_v2.sh" 10 60
        exit 1
    fi
    info "Root check passed"
}

# Get and validate target user
get_target_user() {
    while true; do
        TARGET_USER=$(whiptail --title "Target User" --inputbox "\nEnter the username to configure:" 10 60 3>&1 1>&2 2>&3)
        
        if [[ -z "$TARGET_USER" ]]; then
            if whiptail --title "Confirm" --yesno "\nNo username provided. Exit?" 10 60; then
                exit 0
            else
                continue
            fi
        fi
        
        if id -u "$TARGET_USER" >/dev/null 2>&1; then
            TARGET_HOME=$(eval echo ~$TARGET_USER)
            info "Target user: $TARGET_USER (home: $TARGET_HOME)"
            break
        else
            whiptail --title "Error" --msgbox "\nUser '$TARGET_USER' does not exist." 10 60
        fi
    done
}

# Run command as target user
run_as_user() {
    local cmd="$*"
    sudo -u "$TARGET_USER" bash -c "$cmd"
}

# Check if package is installed
is_installed() {
    local pkg=$1
    dpkg-query -W --showformat='${Status}\n' "$pkg" 2>/dev/null | grep -q "install ok installed"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Show progress with whiptail gauge
show_progress() {
    local title=$1
    local current=$2
    local total=$3
    local item=$4
    
    local percent=$(( (current * 100) / total ))
    echo "XXX"
    echo "$percent"
    echo "$item"
    echo "XXX"
}

# Handle installation result
handle_result() {
    local item=$1
    local result=$2
    
    if [[ $result -eq 0 ]]; then
        COMPLETED_ITEMS+=("$item")
        info "✓ Completed: $item"
    elif [[ $result -eq 99 ]]; then
        SKIPPED_ITEMS+=("$item")
        info "⊘ Skipped: $item (already installed/not available)"
    else
        FAILED_ITEMS+=("$item")
        error "✗ Failed: $item (exit code: $result)"
        
        if whiptail --title "Installation Failed" --yesno "\n$item installation failed.\n\nContinue with remaining items?" 10 60; then
            return 0
        else
            exit 1
        fi
    fi
}

# Show installation summary
show_summary() {
    local summary="Installation Summary:\n\n"
    
    if [[ ${#COMPLETED_ITEMS[@]} -gt 0 ]]; then
        summary+="✓ Completed (${#COMPLETED_ITEMS[@]}):\n"
        for item in "${COMPLETED_ITEMS[@]}"; do
            summary+="  - $item\n"
        done
    fi
    
    summary+="\n"
    
    if [[ ${#SKIPPED_ITEMS[@]} -gt 0 ]]; then
        summary+="⊘ Skipped (${#SKIPPED_ITEMS[@]}):\n"
        for item in "${SKIPPED_ITEMS[@]}"; do
            summary+="  - $item\n"
        done
    fi
    
    summary+="\n"
    
    if [[ ${#FAILED_ITEMS[@]} -gt 0 ]]; then
        summary+="✗ Failed (${#FAILED_ITEMS[@]}):\n"
        for item in "${FAILED_ITEMS[@]}"; do
            summary+="  - $item\n"
        done
    fi
    
    summary+="\n"
    summary+="See detailed log: $LOG_FILE\n"
    
    whiptail --title "Installation Complete" --msgbox "$summary" 20 70
}

#==============================================================================
# CATEGORY 1: CORE SYSTEM & UTILITIES
#==============================================================================

install_core_tools() {
    local items=("git" "curl" "gedit" "nfs-common" "preload")
    local item_names=("Git" "cURL" "Gedit" "NFS Utilities" "Preload")
    local total=${#items[@]}
    local current=0
    
    {
        for i in "${!items[@]}"; do
            local pkg=${items[$i]}
            local name=${item_names[$i]}
            ((current++))
            
            show_progress "Installing Core Tools" "$current" "$total" "$name"
            
            if is_installed "$pkg"; then
                handle_result "$name" 99
            else
                if apt install -y "$pkg" >>"${LOG_FILE}" 2>&1; then
                    handle_result "$name" 0
                else
                    handle_result "$name" 1
                fi
            fi
        done
    } | whiptail --title "Installing Core Tools" --gauge "\nPlease wait..." 10 60 0
}

configure_git() {
    local title="Configure Git"
    
    if git config --global user.name >/dev/null 2>&1 && git config --global user.email >/dev/null 2>&1; then
        if whiptail --title "Git Configuration" --yesno "\nGit is already configured.\n\nReconfigure?" 10 60; then
            info "Reconfiguring Git"
        else
            info "Git configuration skipped"
            return 99
        fi
    fi
    
    local git_name git_email
    
    while true; do
        git_name=$(whiptail --title "$title" --inputbox "Enter Git user.name:" 10 60 3>&1 1>&2 2>&3)
        [[ -n "$git_name" ]] && break
        whiptail --title "Invalid Input" --msgbox "Name cannot be empty." 10 60
    done
    
    while true; do
        git_email=$(whiptail --title "$title" --inputbox "Enter Git user.email:" 10 60 3>&1 1>&2 2>&3)
        [[ -n "$git_email" ]] && break
        whiptail --title "Invalid Input" --msgbox "Email cannot be empty." 10 60
    done
    
    {
        git config --global user.name "$git_name"
        git config --global user.email "$git_email"
        git config --global pager.branch false
        git config --global credential.helper store
        
        handle_result "Git Configuration" 0
    } | whiptail --title "$title" --gauge "\nConfiguring git..." 10 60 0
}

install_system_monitors() {
    local items=("vim" "stacer" "lm-sensors")
    local item_names=("VIM" "Stacer" "lm-sensors")
    local total=${#items[@]}
    local current=0
    
    {
        for i in "${!items[@]}"; do
            local pkg=${items[$i]}
            local name=${item_names[$i]}
            ((current++))
            
            show_progress "Installing System Monitors" "$current" "$total" "$name"
            
            if is_installed "$pkg"; then
                handle_result "$name" 99
            else
                if apt install -y "$pkg" >>"${LOG_FILE}" 2>&1; then
                    if [[ "$pkg" == "lm-sensors" ]]; then
                        sensors-detect --auto >>"${LOG_FILE}" 2>&1
                    fi
                    handle_result "$name" 0
                else
                    handle_result "$name" 1
                fi
            fi
        done
    } | whiptail --title "Installing System Monitors" --gauge "\nPlease wait..." 10 60 0
}

perform_system_upgrade() {
    local title="System Upgrade"
    
    if whiptail --title "$title" --yesno "\nThis will run:\n\n- apt dist-upgrade\n- update-initramfs\n\nThis may take a while. Continue?" 12 60; then
        {
            show_progress "System Upgrade" 1 3 "Updating package lists"
            apt update >>"${LOG_FILE}" 2>&1
            
            show_progress "System Upgrade" 2 3 "Upgrading packages"
            apt dist-upgrade -y >>"${LOG_FILE}" 2>&1
            
            show_progress "System Upgrade" 3 3 "Updating initramfs"
            update-initramfs -u >>"${LOG_FILE}" 2>&1
            
            handle_result "System Upgrade" 0
        } | whiptail --title "$title" --gauge "\nPerforming system upgrade..." 10 60 0
    else
        handle_result "System Upgrade" 99
    fi
}

#==============================================================================
# CATEGORY 2: SECURITY & REMOTE ACCESS
#==============================================================================

install_ssh_tools() {
    local items=("sshpass")
    local item_names=("SSH Tools")
    local total=${#items[@]}
    
    # Check for SSH keys
    local ssh_key_path="${SCRIPT_DIR}/dot.ssh.zip"
    if [[ ! -f "$ssh_key_path" ]]; then
        if whiptail --title "Missing SSH Keys" --yesno "\nSSH key file not found:\n$ssh_key_path\n\nContinue without installing SSH keys?" 12 60; then
            return 99
        else
            whiptail --title "Error" --msgbox "\nSSH key file required. Please add dot.ssh.zip to script directory." 10 60
            return 1
        fi
    fi
    
    # Check for sshuttle
    if is_installed "sshuttle"; then
        if ! whiptail --title "SSH Tools" --yesno "\nSSH tools are already installed.\n\nReinstall to extract keys and configure?" 10 60; then
            return 99
        fi
    fi
    
    {
        show_progress "Installing SSH Tools" 1 3 "Installing sshuttle"
        if apt install -y sshuttle >>"${LOG_FILE}" 2>&1; then
            handle_result "sshuttle" 0
        else
            handle_result "sshuttle" 1
        fi
        
        show_progress "Installing SSH Tools" 2 3 "Configuring SSH"
        if [[ -f "$ssh_key_path" ]]; then
            run_as_user "rm -rf ${TARGET_HOME}/.ssh"
            run_as_user "mkdir -p ${TARGET_HOME}/.ssh"
            run_as_user "unzip -o ${ssh_key_path} -d ${TARGET_HOME}/"
            run_as_user "chmod 700 ${TARGET_HOME}/.ssh"
            run_as_user "chmod 600 ${TARGET_HOME}/.ssh/*"
            handle_result "SSH Keys" 0
        fi
        
        show_progress "Installing SSH Tools" 3 3 "Adding VPN function"
        run_as_user "cat >> ${TARGET_HOME}/.zshrc << 'EOF'

# SSH VPN function
sshuttle_vpn() {
    remoteUsername='user'
    remoteHostname='hostname.com'
    sshuttle --dns --verbose --remote \$remoteUsername@\$remoteHostname --exclude \$remoteHostname 0/0
}
EOF"
        handle_result "VPN Function" 0
    } | whiptail --title "Installing SSH Tools" --gauge "\nConfiguring SSH and VPN..." 10 60 0
}

install_zerotier() {
    local title="ZeroTier Network"
    
    if command_exists "zerotier-cli"; then
        if ! whiptail --title "$title" --yesno "\nZeroTier is already installed.\n\nReinstall/update?" 10 60; then
            return 99
        fi
    fi
    
    {
        show_progress "Installing ZeroTier" 1 2 "Running installer"
        if curl -s https://install.zerotier.com | bash >>"${LOG_FILE}" 2>&1; then
            show_progress "Installing ZeroTier" 2 2 "Enabling service"
            systemctl enable --now zerotier-one >>"${LOG_FILE}" 2>&1
            handle_result "ZeroTier" 0
        else
            handle_result "ZeroTier" 1
        fi
    } | whiptail --title "$title" --gauge "\nInstalling ZeroTier..." 10 60 0
}

install_ssh_server() {
    local title="OpenSSH Server"
    
    if ! whiptail --title "$title" --yesno "\nInstalling OpenSSH Server will allow remote access to this machine.\n\nContinue?" 11 60; then
        return 99
    fi
    
    {
        show_progress "Installing SSH Server" 1 3 "Installing packages"
        if apt install -y openssh-server >>"${LOG_FILE}" 2>&1; then
            show_progress "Installing SSH Server" 2 3 "Enabling service"
            systemctl enable ssh >>"${LOG_FILE}" 2>&1
            show_progress "Installing SSH Server" 3 3 "Starting service"
            systemctl start ssh >>"${LOG_FILE}" 2>&1
            handle_result "SSH Server" 0
        else
            handle_result "SSH Server" 1
        fi
    } | whiptail --title "$title" --gauge "\nInstalling SSH Server..." 10 60 0
}

#==============================================================================
# CATEGORY 3: SHELL ENVIRONMENT
#==============================================================================

install_zsh_environment() {
    local title="ZSH Environment"
    
    if ! whiptail --title "$title" --yesno "\nThis will install:\n\n- ZSH shell\n- Powerline & FiraCode fonts\n- Custom fonts from my-fonts/\n- Oh My Zsh framework\n- Powerlevel10k theme\n- z.lua directory jumper\n\nContinue?" 14 60; then
        return 99
    fi
    
    local total=8
    local current=0
    
    {
        # Install ZSH
        ((current++))
        show_progress "Installing ZSH Environment" "$current" "$total" "ZSH Shell"
        if ! is_installed "zsh"; then
            if apt install -y zsh >>"${LOG_FILE}" 2>&1; then
                handle_result "ZSH Shell" 0
            else
                handle_result "ZSH Shell" 1
            fi
        else
            handle_result "ZSH Shell" 99
        fi
        
        # Set ZSH as default
        ((current++))
        show_progress "Installing ZSH Environment" "$current" "$total" "Setting default shell"
        chsh -s /bin/zsh "$TARGET_USER" >>"${LOG_FILE}" 2>&1
        handle_result "Default Shell" 0
        
        # Install fonts
        ((current++))
        show_progress "Installing ZSH Environment" "$current" "$total" "Powerline fonts"
        if ! is_installed "fonts-powerline"; then
            if apt install -y powerline >>"${LOG_FILE}" 2>&1; then
                handle_result "Powerline" 0
            else
                handle_result "Powerline" 1
            fi
        else
            handle_result "Powerline" 99
        fi
        
        ((current++))
        show_progress "Installing ZSH Environment" "$current" "$total" "FiraCode fonts"
        if ! is_installed "fonts-firacode"; then
            if apt install -y fonts-firacode >>"${LOG_FILE}" 2>&1; then
                handle_result "FiraCode" 0
            else
                handle_result "FiraCode" 1
            fi
        else
            handle_result "FiraCode" 99
        fi
        
        ((current++))
        show_progress "Installing ZSH Environment" "$current" "$total" "Custom fonts"
        if [[ -d "$SCRIPT_DIR/my-fonts" ]]; then
            run_as_user "mkdir -p ${TARGET_HOME}/.fonts"
            run_as_user "find ${SCRIPT_DIR}/my-fonts -type f \( -iname '*.ttf' -o -iname '*.otf' \) -exec cp {} ${TARGET_HOME}/.fonts/ \;"
            run_as_user "fc-cache -fv ${TARGET_HOME}/.fonts" >>"${LOG_FILE}" 2>&1
            handle_result "Custom Fonts" 0
        else
            warn "my-fonts/ directory not found, skipping custom fonts"
            handle_result "Custom Fonts" 99
        fi
        
        # Install Oh My Zsh
        ((current++))
        show_progress "Installing ZSH Environment" "$current" "$total" "Oh My Zsh"
        if [[ -d "${TARG