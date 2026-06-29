#!/bin/bash

set -euo pipefail

VENTOY_DIR="/tmp/opencode"
GITHUB_REPO="ventoy/Ventoy"
LATEST_VERSION=""
INSTALLED_VERSION=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    clear
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}     Ventoy2Disk Manager v1.0        ${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_info()    { echo -e "${BLUE}ℹ $1${NC}"; }

check_sudo() {
    if ! sudo -v &>/dev/null; then
        print_error "This script requires sudo privileges."
        exit 1
    fi
}

section() {
    print_header
    echo -e "${YELLOW}--- $1 ---${NC}"
    echo ""
}

get_latest_version() {
    if [[ -n "$LATEST_VERSION" ]]; then
        echo "$LATEST_VERSION"
        return
    fi
    LATEST_VERSION=$(curl -sL "https://api.github.com/repos/$GITHUB_REPO/releases/latest" \
        | grep -Po '"tag_name":\s*"\K[^"]+' \
        | sed 's/^v//' || true)
    echo "$LATEST_VERSION"
}

get_installed_version() {
    local disk="${1:-$SELECTED_DISK}"
    local ver_dir
    ver_dir=$(ls -d "$VENTOY_DIR"/ventoy-* 2>/dev/null | sort -V | tail -1)
    [[ -z "$ver_dir" || ! -f "$ver_dir/Ventoy2Disk.sh" ]] && { echo "unknown"; return; }
    run_ventoy2disk "$ver_dir" -l "$disk" 2>/dev/null \
        | grep -Po 'Ventoy Version in Disk:\s*\K.*' || echo "not installed"
}

check_tool_deps() {
    if ! command -v mkexfatfs &>/dev/null && ! command -v mkfs.exfat &>/dev/null; then
        print_warning "exfatprogs not found — installing..."
        sudo apt install -y exfatprogs
        sudo ln -sf /usr/sbin/mkfs.exfat /usr/sbin/mkexfatfs
    elif ! command -v mkexfatfs &>/dev/null && command -v mkfs.exfat &>/dev/null; then
        sudo ln -sf /usr/sbin/mkfs.exfat /usr/sbin/mkexfatfs
    fi
    print_success "Dependencies satisfied."
}

select_disk() {
    echo ""
    print_info "Available disks:"
    echo ""
    lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,MODEL | grep -E 'disk' | nl -w2 -s') '
    echo ""
    read -p "Select disk number: " disk_num
    SELECTED_DISK=$(lsblk -ndo NAME | sed -n "${disk_num}p")
    if [[ -z "$SELECTED_DISK" ]]; then
        print_error "Invalid selection."
        return 1
    fi
    SELECTED_DISK="/dev/$SELECTED_DISK"
    print_info "Selected: $SELECTED_DISK"
    echo ""
}

confirm_disk() {
    local action="$1"
    echo ""
    lsblk "$SELECTED_DISK" -o NAME,SIZE,TYPE,MOUNTPOINT 2>/dev/null
    echo ""
    print_warning "This will $action ALL data on $SELECTED_DISK"
    read -p "Are you sure? (y/N): " ans
    [[ "$ans" =~ ^[Yy]$ ]]
}

run_ventoy2disk() {
    local ver_dir="$1"
    shift
    (cd "$ver_dir" && sudo sh Ventoy2Disk.sh "$@")
}

download_ventoy() {
    local version="$1"
    local tarball="ventoy-${version}-linux.tar.gz"
    local url="https://github.com/$GITHUB_REPO/releases/download/v${version}/$tarball"

    mkdir -p "$VENTOY_DIR"
    if [[ -d "$VENTOY_DIR/ventoy-$version" ]]; then
        print_info "Ventoy $version already downloaded."
        return 0
    fi

    print_info "Downloading Ventoy ${version}..."
    if ! curl -L -o "$VENTOY_DIR/$tarball" "$url"; then
        print_error "Download failed."
        return 1
    fi

    print_info "Extracting..."
    tar -xzf "$VENTOY_DIR/$tarball" -C "$VENTOY_DIR"
    rm -f "$VENTOY_DIR/$tarball"
    print_success "Downloaded Ventoy $version."
}

install_ventoy() {
    section "Install Ventoy"

    select_disk || return
    confirm_disk "INSTALL Ventoy on" || { print_info "Cancelled."; return; }

    check_tool_deps

    local version
    version=$(get_latest_version)
    [[ -z "$version" ]] && { print_error "Could not fetch latest version."; return 1; }

    download_ventoy "$version" || return

    print_info "Partition style:"
    echo "  1) MBR (default)"
    echo "  2) GPT"
    read -p "Choose [1-2]: " gpt_choice

    local gpt_flag=""
    [[ "$gpt_choice" == "2" ]] && gpt_flag="-g"

    print_info "Installing Ventoy $version with${gpt_flag:- MBR}..."
    run_ventoy2disk "$VENTOY_DIR/ventoy-$version" -I $gpt_flag "$SELECTED_DISK" <<< $'y\ny'
    print_success "Ventoy $version installed on $SELECTED_DISK!"
}

update_ventoy() {
    section "Update Ventoy"

    select_disk || return

    local version
    version=$(get_latest_version)
    [[ -z "$version" ]] && { print_error "Could not fetch latest version."; return 1; }

    download_ventoy "$version" || return
    local current_ver
    current_ver=$(get_installed_version "$SELECTED_DISK")

    if [[ -z "$current_ver" ]]; then
        print_error "No Ventoy installation found on $SELECTED_DISK. Install it first."
        return 1
    fi

    print_info "Current: $current_ver → Latest: $version"

    if [[ "$current_ver" == "$version" ]]; then
        print_info "Already up to date."
        return 0
    fi

    download_ventoy "$version" || return

    print_warning "Updating Ventoy on $SELECTED_DISK from $current_ver to $version"
    read -p "Proceed? (y/N): " ans
    [[ "$ans" =~ ^[Yy]$ ]] || { print_info "Cancelled."; return; }

    run_ventoy2disk "$VENTOY_DIR/ventoy-$version" -u "$SELECTED_DISK"
    print_success "Ventoy updated to $version!"
}

show_status() {
    section "Ventoy Status"

    local version
    version=$(get_latest_version)
    if [[ -n "$version" ]]; then
        print_info "Latest Ventoy: $version"
    else
        print_warning "Could not check latest version."
    fi
    echo ""

    print_info "Available disks:"
    echo ""
    lsblk -o NAME,SIZE,TYPE,MODEL | grep -E 'disk'
    echo ""
    read -p "Enter disk to inspect (e.g., sda): " disk_name
    [[ -z "$disk_name" ]] && { print_info "Skipped."; return; }

    local disk="/dev/$disk_name"
    if [[ ! -b "$disk" ]]; then
        print_error "$disk not found."
        return
    fi

    download_ventoy "$version" 2>/dev/null || true
    local ver_dir
    ver_dir=$(ls -d "$VENTOY_DIR"/ventoy-* 2>/dev/null | sort -V | tail -1)
    if [[ -z "$ver_dir" || ! -f "$ver_dir/Ventoy2Disk.sh" ]]; then
        print_error "No Ventoy release downloaded. Check internet."
        return
    fi
    local ver
    ver=$(run_ventoy2disk "$ver_dir" -l "$disk" 2>/dev/null || echo "")
    if [[ -n "$ver" ]]; then
        echo ""
        echo "$ver"
    else
        print_error "No Ventoy installation detected on $disk."
    fi
}

main_menu() {
    if [[ ! -t 0 ]]; then
        case "${1:-}" in
            1|install) install_ventoy ;;
            2|update)  update_ventoy ;;
            3|status)  show_status ;;
            *)         show_status ;;
        esac
        return
    fi

    while true; do
        print_header
        cat <<EOF
Please select an option:

  1) Install Ventoy (fresh install, MBR or GPT)
  2) Update Ventoy (safe upgrade, keeps data)
  3) Show Status (check disk for Ventoy)
  4) Exit

EOF
        read -p "Enter your choice [1-4]: " choice

        case $choice in
            1) install_ventoy ;;
            2) update_ventoy ;;
            3) show_status ;;
            4) print_header; print_success "Goodbye!"; exit 0 ;;
            *) print_error "Invalid option."; sleep 1 ;;
        esac

        echo ""
        read -p "Press Enter to continue..." || true
    done
}

main_menu "${1:-}"
