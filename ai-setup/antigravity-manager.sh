#!/bin/bash

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

DOWNLOAD_PAGE="https://antigravity.google/download"
AGY_INSTALLER_URL="https://antigravity.google/cli/install.sh"

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}   Antigravity Manager${NC}"
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

check_desktop_installed() {
    if [ -x /usr/local/bin/antigravity ] && [ -d /opt/antigravity ]; then
        return 0
    fi
    return 1
}

check_ide_installed() {
    if [ -x /usr/local/bin/antigravity-ide ] && [ -d /opt/antigravity-ide ]; then
        return 0
    fi
    return 1
}

check_cli_installed() {
    if command -v agy &> /dev/null; then
        return 0
    fi
    return 1
}

get_desktop_version() {
    cat /opt/antigravity/.linuxcapable-version 2>/dev/null || echo "?"
}

get_ide_version() {
    cat /opt/antigravity-ide/.linuxcapable-version 2>/dev/null || echo "?"
}

get_cli_version() {
    agy --version 2>/dev/null || echo "?"
}

install_desktop() {
    print_status "Installing Antigravity 2.0 Desktop App..."

    for cmd in curl tar python3; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "$cmd is required. Install with: sudo apt install $cmd"
            return 1
        fi
    done

    if check_desktop_installed; then
        print_warning "Desktop app is already installed!"
        read -p "Reinstall? (y/N): " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then return; fi
    fi

    if [ ! -f /usr/local/bin/update-antigravity ]; then
        print_status "Creating update-antigravity helper..."
        sudo tee /usr/local/bin/update-antigravity > /dev/null <<'HELPER'
#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then echo "Run with sudo" >&2; exit 1; fi
download_page="${1:-https://antigravity.google/download}"
install_root="/opt/antigravity"; command_link="/usr/local/bin/antigravity"
desktop_file="/usr/share/applications/antigravity.desktop"
icon_file="/usr/share/icons/hicolor/512x512/apps/antigravity.png"
case "$(uname -m)" in x86_64|amd64) platform="linux-x64" ;; aarch64|arm64) platform="linux-arm" ;; *) echo "Unsupported" >&2; exit 1 ;; esac
for cmd in curl tar python3; do command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd required" >&2; exit 1; }; done
if [ -L "$command_link" ]; then t=$(readlink -f "$command_link" || true); case "$t" in "$install_root"/*) ;; *) echo "Bad link" >&2; exit 1 ;; esac; elif [ -e "$command_link" ]; then echo "Exists" >&2; exit 1; fi
tmpdir=$(mktemp -d "${TMPDIR:-/var/tmp}/antigravity.XXXXXX"); trap 'rm -rf "$tmpdir"' EXIT
curl -fsSL --compressed --retry 3 -o "$tmpdir/dl.html" "$download_page"
main_js=$(python3 - "$tmpdir/dl.html" "$download_page" <<'PY'
import re,sys;from urllib.parse import urljoin
h=open(sys.argv[1]).read();m=re.findall(r'(?:src|href)="([^"]*main-[^"]+\.js)"',h)
print(urljoin(sys.argv[2],m[-1]))
PY
)
curl -fsSL --compressed --retry 3 -o "$tmpdir/dl.js" "$main_js"
IFS=' ' read -r ver url <<<"$(python3 - "$tmpdir/dl.js" "$platform" <<'PY'
import re,sys
b=open(sys.argv[1],errors='replace').read();p=sys.argv[2];s=b.find('id:"antigravity-2"');e=b.find('},{name:"command",id:"antigravity-cli"',s)
m=re.search(r'href:"([^"]+/'+re.escape(p)+r'/Antigravity\.tar\.gz)"',b[s:e])
vm=re.search(r'/antigravity-hub/([^/]+)/',m.group(1))
print(vm.group(1).split("-",1)[0],m.group(1))
PY
)"
[ -n "$ver" ] && [ -n "$url" ] || { echo "Parse failed" >&2; exit 1; }
case "$platform" in linux-x64) top="Antigravity-x64" ;; linux-arm) top="Antigravity-arm64" ;; esac
[ -f "$icon_file" ] || icon_old=1
if [ "$(cat "$install_root/.linuxcapable-version" 2>/dev/null)" = "$ver" ] && [ -x "$install_root/$top/antigravity" ] && [ -f "$desktop_file" ] && [ -f "$icon_file" ]; then printf 'Already installed: %s\n' "$ver"; exit 0; fi
echo "Downloading $ver..."; curl -fsSL --retry 3 -o "$tmpdir/arc.tar.gz" "$url"
tar -xzf "$tmpdir/arc.tar.gz" -C "$tmpdir"; [ -x "$tmpdir/$top/antigravity" ] || { echo "Binary not found" >&2; exit 1; }
python3 - "$tmpdir/$top/resources/app.asar" "$tmpdir/icon.png" <<'PY2'
import json,struct,sys
a=open(sys.argv[1],"rb");o=open(sys.argv[2],"wb");a.read(4);hs=struct.unpack("<I",a.read(4))[0];a.read(4);js=struct.unpack("<I",a.read(4))[0]
h=json.loads(a.read(js).decode());ic=h["files"]["icon.png"];a.seek(8+hs+int(ic["offset"]));o.write(a.read(int(ic["size"])))
PY2
rm -rf "${install_root}.new"; mkdir -p "${install_root}.new"; cp -a "$tmpdir/$top" "${install_root}.new/"; echo "$ver" >"${install_root}.new/.linuxcapable-version"
if [ -f "${install_root}.new/$top/chrome-sandbox" ]; then chown root:root "${install_root}.new/$top/chrome-sandbox"; chmod 4755 "${install_root}.new/$top/chrome-sandbox"; fi
[ -d "$install_root" ] && { rm -rf "${install_root}.previous"; mv "$install_root" "${install_root}.previous"; }
mv "${install_root}.new" "$install_root"; ln -sfn "$install_root/$top/antigravity" "$command_link"
mkdir -p "$(dirname "$icon_file")"; install -m 0644 "$tmpdir/icon.png" "$icon_file"
tee "$desktop_file" >/dev/null <<DESKTOP
[Desktop Entry]
Name=Antigravity
Comment=Google Antigravity 2.0
Exec=$command_link %U
Icon=antigravity
Terminal=false
Type=Application
Categories=Development;IDE;
StartupNotify=true
StartupWMClass=Antigravity
DESKTOP
update-desktop-database /usr/share/applications 2>/dev/null || true
gtk-update-icon-cache -q /usr/share/icons/hicolor 2>/dev/null || true
printf 'Installed Antigravity %s\n' "$ver"
HELPER
        sudo chmod +x /usr/local/bin/update-antigravity
    fi

    sudo /usr/local/bin/update-antigravity && print_status "Desktop app installation complete!" || print_error "Desktop app installation failed."
}

install_ide() {
    print_status "Installing Antigravity IDE..."

    for cmd in curl tar python3; do
        if ! command -v "$cmd" &> /dev/null; then
            print_error "$cmd is required. Install with: sudo apt install $cmd"
            return 1
        fi
    done

    if check_ide_installed; then
        print_warning "IDE is already installed!"
        read -p "Reinstall? (y/N): " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then return; fi
    fi

    if [ ! -f /usr/local/bin/update-antigravity-ide ]; then
        print_status "Creating update-antigravity-ide helper..."
        sudo tee /usr/local/bin/update-antigravity-ide > /dev/null <<'HELPER'
#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -ne 0 ]; then echo "Run with sudo" >&2; exit 1; fi
download_page="${1:-https://antigravity.google/download}"
install_root="/opt/antigravity-ide"; command_link="/usr/local/bin/antigravity-ide"
desktop_file="/usr/share/applications/antigravity-ide.desktop"
icon_file="/usr/share/icons/hicolor/512x512/apps/antigravity-ide.png"
case "$(uname -m)" in x86_64|amd64) platform="linux-x64" ;; aarch64|arm64) platform="linux-arm" ;; *) echo "Unsupported" >&2; exit 1 ;; esac
for cmd in curl tar python3; do command -v "$cmd" >/dev/null 2>&1 || { echo "$cmd required" >&2; exit 1; }; done
if [ -L "$command_link" ]; then t=$(readlink -f "$command_link" || true); case "$t" in "$install_root"/*) ;; *) echo "Bad link" >&2; exit 1 ;; esac; elif [ -e "$command_link" ]; then echo "Exists" >&2; exit 1; fi
tmpdir=$(mktemp -d "${TMPDIR:-/var/tmp}/antigravity-ide.XXXXXX"); trap 'rm -rf "$tmpdir"' EXIT
curl -fsSL --compressed --retry 3 -o "$tmpdir/dl.html" "$download_page"
main_js=$(python3 - "$tmpdir/dl.html" "$download_page" <<'PY'
import re,sys;from urllib.parse import urljoin
h=open(sys.argv[1]).read();m=re.findall(r'(?:src|href)="([^"]*main-[^"]+\.js)"',h)
print(urljoin(sys.argv[2],m[-1]))
PY
)
curl -fsSL --compressed --retry 3 -o "$tmpdir/dl.js" "$main_js"
IFS=' ' read -r ver url <<<"$(python3 - "$tmpdir/dl.js" "$platform" <<'PY'
import re,sys
b=open(sys.argv[1],errors='replace').read();p=sys.argv[2];s=b.find('id:"antigravity-ide"');e=b.find('},{name:"download",id:"antigravity-sdk"',s)
m=re.search(r'href:"([^"]+/'+re.escape(p)+r'/Antigravity%20IDE\.tar\.gz)"',b[s:e])
vm=re.search(r'/stable/([^/]+)/',m.group(1))
print(vm.group(1).split("-",1)[0],m.group(1))
PY
)"
[ -n "$ver" ] && [ -n "$url" ] || { echo "Parse failed" >&2; exit 1; }
[ "$(cat "$install_root/.linuxcapable-version" 2>/dev/null)" = "$ver" ] && [ -x "$install_root/Antigravity-IDE/antigravity-ide" ] && [ -f "$desktop_file" ] && [ -f "$icon_file" ] && { printf 'Already installed: %s\n' "$ver"; exit 0; }
echo "Downloading IDE $ver..."; curl -fsSL --retry 3 -o "$tmpdir/arc.tar.gz" "$url"
tar -xzf "$tmpdir/arc.tar.gz" -C "$tmpdir"; [ -x "$tmpdir/Antigravity IDE/antigravity-ide" ] || { echo "Binary not found" >&2; exit 1; }
[ -f "$tmpdir/Antigravity IDE/resources/app/resources/linux/code.png" ] || { echo "Icon not found" >&2; exit 1; }
rm -rf "${install_root}.new"; mkdir -p "${install_root}.new/Antigravity-IDE"
cp -a "$tmpdir/Antigravity IDE/." "${install_root}.new/Antigravity-IDE/"; echo "$ver" >"${install_root}.new/.linuxcapable-version"
if [ -f "${install_root}.new/Antigravity-IDE/chrome-sandbox" ]; then chown root:root "${install_root}.new/Antigravity-IDE/chrome-sandbox"; chmod 4755 "${install_root}.new/Antigravity-IDE/chrome-sandbox"; fi
[ -d "$install_root" ] && { rm -rf "${install_root}.previous"; mv "$install_root" "${install_root}.previous"; }
mv "${install_root}.new" "$install_root"; ln -sfn "$install_root/Antigravity-IDE/antigravity-ide" "$command_link"
mkdir -p "$(dirname "$icon_file")"; install -m 0644 "$tmpdir/Antigravity IDE/resources/app/resources/linux/code.png" "$icon_file"
tee "$desktop_file" >/dev/null <<DESKTOP
[Desktop Entry]
Name=Antigravity IDE
Comment=Google Antigravity IDE
Exec=$command_link %U
Icon=antigravity-ide
Terminal=false
Type=Application
Categories=Development;IDE;
MimeType=x-scheme-handler/antigravity-ide;application/x-antigravity-workspace;
StartupNotify=true
StartupWMClass=antigravity-ide
DESKTOP
update-desktop-database /usr/share/applications 2>/dev/null || true
gtk-update-icon-cache -q /usr/share/icons/hicolor 2>/dev/null || true
printf 'Installed Antigravity IDE %s\n' "$ver"
HELPER
        sudo chmod +x /usr/local/bin/update-antigravity-ide
    fi

    sudo /usr/local/bin/update-antigravity-ide && print_status "IDE installation complete!" || print_error "IDE installation failed."
}

install_cli() {
    print_status "Installing Antigravity CLI (agy)..."
    if check_cli_installed; then
        print_warning "CLI is already installed!"
        read -p "Reinstall? (y/N): " -n 1 -r; echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then return; fi
    fi
    curl -fsSL "$AGY_INSTALLER_URL" | bash
    if command -v agy &> /dev/null; then
        print_status "CLI installed: $(agy --version 2>/dev/null || echo '?')"
    else
        print_warning "Installation ran but 'agy' not found. Add ~/.local/bin to PATH or open a new terminal."
    fi
}

install_all() {
    install_desktop
    echo ""
    install_ide
    echo ""
    install_cli
}

uninstall_desktop() {
    print_status "Uninstalling Antigravity 2.0 Desktop App..."
    if [ -L /usr/local/bin/antigravity ] && readlink /usr/local/bin/antigravity 2>/dev/null | grep -q '^/opt/antigravity/'; then
        sudo rm -f /usr/local/bin/antigravity
    fi
    sudo rm -rf /opt/antigravity /opt/antigravity.previous /opt/antigravity.new
    sudo rm -f /usr/local/bin/update-antigravity \
        /usr/share/applications/antigravity.desktop \
        /usr/share/applications/antigravity-x11.desktop \
        /usr/share/icons/hicolor/512x512/apps/antigravity.png \
        /usr/share/icons/hicolor/scalable/apps/antigravity.svg
    sudo update-desktop-database /usr/share/applications 2>/dev/null || true
    sudo gtk-update-icon-cache -q /usr/share/icons/hicolor 2>/dev/null || true
    hash -r
    print_status "Desktop app removed."
}

uninstall_ide() {
    print_status "Uninstalling Antigravity IDE..."
    sudo rm -rf /opt/antigravity-ide /opt/antigravity-ide.previous /opt/antigravity-ide.new
    sudo rm -f /usr/local/bin/antigravity-ide \
        /usr/local/bin/update-antigravity-ide \
        /usr/share/applications/antigravity-ide.desktop \
        /usr/share/applications/antigravity-ide-x11.desktop \
        /usr/share/icons/hicolor/512x512/apps/antigravity-ide.png
    if [ -f /etc/apparmor.d/antigravity-ide ]; then
        sudo apparmor_parser -R /etc/apparmor.d/antigravity-ide 2>/dev/null || true
    fi
    sudo rm -f /etc/apparmor.d/antigravity-ide /etc/apparmor.d/local/antigravity-ide
    sudo update-desktop-database /usr/share/applications 2>/dev/null || true
    sudo gtk-update-icon-cache -q /usr/share/icons/hicolor 2>/dev/null || true
    hash -r
    print_status "IDE removed."
}

uninstall_cli() {
    print_status "Uninstalling Antigravity CLI..."
    rm -f "$HOME/.local/bin/agy" "$HOME/.local/bin/update-antigravity-cli"
    hash -r
    if ! command -v agy &> /dev/null; then
        print_status "CLI removed."
    else
        print_warning "agy still found at $(command -v agy). Remove it manually."
    fi
}

uninstall_all() {
    print_warning "This will remove ALL Antigravity components!"
    read -p "Are you sure? (y/N): " -n 1 -r; echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        uninstall_desktop
        uninstall_ide
        uninstall_cli
        print_status "All components removed."
    else
        print_status "Cancelled."
    fi
}

cache_clear() {
    print_status "Clearing Antigravity caches..."
    local count=0
    local dirs=(
        "$HOME/.cache/antigravity"
        "$HOME/.cache/Antigravity"
        "$HOME/.config/Antigravity/Cache"
        "$HOME/.config/Antigravity/Code Cache"
        "$HOME/.config/Antigravity/DawnCache"
        "$HOME/.config/Antigravity/CachedData"
        "$HOME/.config/Antigravity/CachedExtension"
        "$HOME/.cache/antigravity-ide"
        "$HOME/.cache/Antigravity IDE"
        "$HOME/.config/Antigravity IDE/Cache"
        "$HOME/.config/Antigravity IDE/Code Cache"
        "$HOME/.config/agy"
    )
    for d in "${dirs[@]}"; do
        if [ -d "$d" ]; then
            rm -rf "$d" 2>/dev/null && print_status "Removed: $d" && count=$((count + 1))
        fi
    done
    if [ "$count" -eq 0 ]; then
        print_status "No caches found."
    else
        print_status "Cleared $count cache location(s)."
    fi
}

update_desktop() {
    if [ -f /usr/local/bin/update-antigravity ]; then
        print_status "Updating Antigravity Desktop App..."
        sudo /usr/local/bin/update-antigravity
    else
        print_warning "Desktop not installed. Install it first."
    fi
}

update_ide() {
    if [ -f /usr/local/bin/update-antigravity-ide ]; then
        print_status "Updating Antigravity IDE..."
        sudo /usr/local/bin/update-antigravity-ide
    else
        print_warning "IDE not installed. Install it first."
    fi
}

update_cli() {
    if check_cli_installed; then
        print_status "Updating Antigravity CLI..."
        agy update
    else
        print_warning "CLI not installed. Install it first."
    fi
}

show_menu() {
    print_header

    echo -e "${CYAN}Installed Components:${NC}"
    if check_desktop_installed; then
        echo -e "  Desktop App : ${GREEN}installed${NC} ($(get_desktop_version))"
    else
        echo -e "  Desktop App : ${RED}not installed${NC}"
    fi
    if check_ide_installed; then
        echo -e "  IDE         : ${GREEN}installed${NC} ($(get_ide_version))"
    else
        echo -e "  IDE         : ${RED}not installed${NC}"
    fi
    if check_cli_installed; then
        echo -e "  CLI (agy)   : ${GREEN}installed${NC} ($(get_cli_version))"
    else
        echo -e "  CLI (agy)   : ${RED}not installed${NC}"
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
        echo "1) Desktop App"
        echo "2) IDE"
        echo "3) CLI (agy)"
        echo "4) All components"
        echo "5) Back to main menu"
        echo ""
        read -p "Please enter your choice (1-5): " choice
        case $choice in
            1) install_desktop ;;
            2) install_ide ;;
            3) install_cli ;;
            4) install_all ;;
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
        echo "1) Desktop App"
        echo "2) IDE"
        echo "3) CLI (agy)"
        echo "4) All components"
        echo "5) Back to main menu"
        echo ""
        read -p "Please enter your choice (1-5): " choice
        case $choice in
            1)
                if check_desktop_installed; then
                    read -p "Remove desktop app? (y/N): " -n 1 -r; echo
                    [[ $REPLY =~ ^[Yy]$ ]] && uninstall_desktop
                else
                    print_warning "Desktop app is not installed."
                fi
                ;;
            2)
                if check_ide_installed; then
                    read -p "Remove IDE? (y/N): " -n 1 -r; echo
                    [[ $REPLY =~ ^[Yy]$ ]] && uninstall_ide
                else
                    print_warning "IDE is not installed."
                fi
                ;;
            3)
                if check_cli_installed; then
                    read -p "Remove CLI? (y/N): " -n 1 -r; echo
                    [[ $REPLY =~ ^[Yy]$ ]] && uninstall_cli
                else
                    print_warning "CLI is not installed."
                fi
                ;;
            4) uninstall_all ;;
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
        echo "1) Desktop App"
        echo "2) IDE"
        echo "3) CLI (agy)"
        echo "4) All components"
        echo "5) Back to main menu"
        echo ""
        read -p "Please enter your choice (1-5): " choice
        case $choice in
            1) update_desktop ;;
            2) update_ide ;;
            3) update_cli ;;
            4)
                update_desktop
                update_ide
                update_cli
                ;;
            5) return ;;
            *) print_error "Invalid choice!" ;;
        esac
        echo ""
        read -p "Press Enter to continue..."
    done
}

check_status() {
    print_status "Checking Antigravity status..."
    echo ""
    if check_desktop_installed; then
        echo -e "  Desktop App : ${GREEN}installed${NC}"
        echo -e "    Version    : $(get_desktop_version)"
        echo -e "    Location   : $(readlink -f /usr/local/bin/antigravity)"
    else
        echo -e "  Desktop App : ${RED}not installed${NC}"
    fi
    echo ""
    if check_ide_installed; then
        echo -e "  IDE         : ${GREEN}installed${NC}"
        echo -e "    Version    : $(get_ide_version)"
        echo -e "    Location   : $(readlink -f /usr/local/bin/antigravity-ide)"
    else
        echo -e "  IDE         : ${RED}not installed${NC}"
    fi
    echo ""
    if check_cli_installed; then
        echo -e "  CLI (agy)   : ${GREEN}installed${NC}"
        echo -e "    Version    : $(get_cli_version)"
        echo -e "    Location   : $(command -v agy)"
    else
        echo -e "  CLI (agy)   : ${RED}not installed${NC}"
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
            4) cache_clear ;;
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
