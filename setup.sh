#!/bin/bash
set -euo pipefail

NC='\033[0m'
BOLD='\033[1m'
RED='\033[0;31m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'

LOG_FILE="/tmp/ubuntu-setup-$(date +%Y%m%d-%H%M%S).log"
log_ok()   { echo -e "${GREEN}✅ $1${NC}"; echo "[OK] $1" >> "$LOG_FILE"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; echo "[WARN] $1" >> "$LOG_FILE"; }
log_err()  { echo -e "${RED}❌ $1${NC}"; echo "[ERROR] $1" >> "$LOG_FILE"; }

section() {
  echo ""
  echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}  $1${NC}"
  echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
  echo ""
}

echo -e "${BOLD}${GREEN}🚀 Starting Ultimate Ubuntu Dev Setup${NC}"
echo -e "${YELLOW}Sections: User Config → System Update (mandatory) → Essentials → Drivers → Touchpad → Docker → Fonts → Zsh → Node → Python → Terminal → Chrome → VS Code → Git → FDM → Teams → Optional Utilities${NC}"

# ═══════════════════════════════════════════════════
# 👤 SECTION 1: User Configuration
# ═══════════════════════════════════════════════════
section "👤 User Configuration"

if [[ -z "${SUDO_USER:-}" ]]; then
    while true; do
        echo -e "${BOLD}Enter username for this setup:${NC} "
        read -r SETUP_USER
        if id "$SETUP_USER" &>/dev/null; then
            break
        fi
        echo -e "${BOLD}❌ User '$SETUP_USER' does not exist. Try again.${NC}"
    done
else
    SETUP_USER="$SUDO_USER"
fi

USER_HOME="/home/$SETUP_USER"
ZSHRC="$USER_HOME/.zshrc"

echo -e "${BOLD}Switch to zsh as default shell? (y/n)${NC}"
read -r SWITCH_TO_ZSH

# ═══════════════════════════════════════════════════
# 🔄 SECTION 2: System Update
# ═══════════════════════════════════════════════════
section "🔄 Updating system packages"

sudo apt update && sudo apt upgrade -y || true

# ═══════════════════════════════════════════════════
# 🧩 SECTION 3: Essential Packages
# ═══════════════════════════════════════════════════
section "🧩 Installing essential packages"

echo -e "${BOLD}Install essential packages? (y/n)${NC}"
read -r INSTALL_ESSENTIALS

if [[ "$INSTALL_ESSENTIALS" =~ ^[Yy]$ ]]; then
sudo apt install -y \
  build-essential curl wget git unzip tmux \
  software-properties-common apt-transport-https \
  ca-certificates gnupg lsb-release \
  net-tools dconf-cli fonts-powerline \
  xclip xsel vim neovim vlc || true
else
  echo -e "${YELLOW}⏭️ Skipping essential packages${NC}"
fi

# ═══════════════════════════════════════════════════
# 💻 SECTION 4: Drivers
# ═══════════════════════════════════════════════════
section "💻 Installing drivers"

echo -e "${BOLD}Install drivers? (y/n)${NC}"
read -r INSTALL_DRIVERS

if [[ "$INSTALL_DRIVERS" =~ ^[Yy]$ ]]; then
sudo ubuntu-drivers autoinstall || true
else
  echo -e "${YELLOW}⏭️ Skipping drivers${NC}"
fi

# ═══════════════════════════════════════════════════
# 🖱️ SECTION 5: Touchpad fix for ELAN1300 I2C HID
# ═══════════════════════════════════════════════════
section "🖱️ Touchpad fix for ELAN1300 I2C HID"

echo -e "${BOLD}Apply touchpad fix for ELAN1300 I2C HID? (y/n)${NC}"
read -r INSTALL_TOUCHPAD

if [[ "$INSTALL_TOUCHPAD" =~ ^[Yy]$ ]]; then

# ───────────────────────────────────────────────────
# 1. Kernel parameter fix (safe + robust)
# ───────────────────────────────────────────────────
if grep -q "^GRUB_CMDLINE_LINUX_DEFAULT=" /etc/default/grub; then
  if ! grep -q "i2c_hid.reset_descriptor=1" /etc/default/grub; then
    sudo sed -i 's/\(GRUB_CMDLINE_LINUX_DEFAULT=".*\)"/\1 i2c_hid.reset_descriptor=1"/' /etc/default/grub
    sudo update-grub || true
    echo -e "${GREEN}✅ I2C HID kernel parameter added (reboot required)${NC}"
  else
    echo -e "${GREEN}✅ I2C HID kernel parameter already set${NC}"
  fi
fi

# ───────────────────────────────────────────────────
# 2. Targeted udev rule (ONLY ELAN1300)
# ───────────────────────────────────────────────────
sudo tee /etc/udev/rules.d/99-touchpad-fix.rules > /dev/null << "EOF"
# ────────────────────────────────────────────────────────────────────────────
# 🎯 PICK ONE RULE BELOW (uncomment only one):
# ────────────────────────────────────────────────────────────────────────────

# ✅ RECOMMENDED: Target ELAN1300 touchpad ONLY (no side effects)
ACTION=="add", SUBSYSTEM=="i2c", ATTR{name}=="ELAN1300*", ATTR{power/control}="on"

# 🟡 ALTERNATIVE: All I2C HID devices (for other touchpad models)
# ACTION=="add", SUBSYSTEM=="i2c", DRIVER=="i2c_hid", ATTR{power/control}="on"
# ACTION=="add", SUBSYSTEM=="i2c", DRIVER=="i2c_hid_acpi", ATTR{power/control}="on"

# 🔴 BROAD: All I2C bus devices (last resort if nothing else works)
# ACTION=="add", SUBSYSTEM=="hid", KERNELS=="i2c-*", ATTR{power/control}="on"
EOF

sudo udevadm control --reload-rules 2>/dev/null || true
sudo udevadm trigger 2>/dev/null || true
echo -e "${GREEN}✅ Targeted udev rule created for ELAN1300${NC}"

# ───────────────────────────────────────────────────
# 3. Suspend/resume fix (only if not exists)
# ───────────────────────────────────────────────────
if [ ! -f /lib/systemd/system-sleep/touchpad-fix.sh ]; then
  sudo tee /lib/systemd/system-sleep/touchpad-fix.sh > /dev/null << "EOF"
#!/bin/bash
# Reload touchpad driver after waking from suspend
[ "$1" = "post" ] && {
  modprobe -r i2c_hid_acpi 2>/dev/null || true
  modprobe -r i2c_hid      2>/dev/null || true
  modprobe    i2c_hid      2>/dev/null || true
  modprobe    i2c_hid_acpi 2>/dev/null || true
}
EOF
  sudo chmod +x /lib/systemd/system-sleep/touchpad-fix.sh
  echo -e "${GREEN}✅ Suspend/resume hook installed${NC}"
else
  echo -e "${GREEN}✅ Suspend/resume hook already exists${NC}"
fi

# ───────────────────────────────────────────────────
# 4. Cleanup old conflicting configs
# ───────────────────────────────────────────────────
# Remove conflicting files from old touchpad setups
sudo rm -f /etc/udev/rules.d/99-touchpad-no-autosuspend.rules 2>/dev/null || true
sudo rm -f /lib/systemd/system-sleep/touchpad-resume.sh 2>/dev/null || true
sudo rm -f /etc/systemd/system/touchpad-persist.service 2>/dev/null || true

sudo systemctl daemon-reload 2>/dev/null || true

echo -e "${GREEN}✅ Touchpad fix applied (ELAN1300 targeted)${NC}"
echo -e "${YELLOW}⚠️  Reboot your system for all changes to take effect${NC}"
else
  echo -e "${YELLOW}⏭️ Skipping touchpad fix${NC}"
fi

# ═══════════════════════════════════════════════════
# 🐳 SECTION 6: Docker
# ═══════════════════════════════════════════════════
section "🐳 Installing Docker"

echo -e "${BOLD}Install Docker? (y/n)${NC}"
read -r INSTALL_DOCKER

if [[ "$INSTALL_DOCKER" =~ ^[Yy]$ ]]; then
sudo apt install -y docker.io || true
sudo apt install -y docker-compose-plugin 2>/dev/null || sudo apt install -y docker-compose || true
sudo systemctl enable docker || true
sudo usermod -aG docker "$SETUP_USER" || true
else
  echo -e "${YELLOW}⏭️ Skipping Docker${NC}"
fi

# ═══════════════════════════════════════════════════
# 🔤 SECTION 7: Fonts
# ═══════════════════════════════════════════════════
section "🔤 Installing FiraCode Nerd Font"

echo -e "${BOLD}Install FiraCode Nerd Font? (y/n)${NC}"
read -r INSTALL_FONTS

if [[ "$INSTALL_FONTS" =~ ^[Yy]$ ]]; then
FONT_DIR="$USER_HOME/.local/share/fonts"
mkdir -p "$FONT_DIR"

(
cd /tmp || exit

FONT_URLS=(
  "https://github.com/ryanoasis/nerd-fonts/raw/master/patched-fonts/FiraCode/Regular/FiraCodeNerdFont-Regular.ttf"
  "https://github.com/ryanoasis/nerd-fonts/raw/master/patched-fonts/FiraCode/Medium/FiraCodeNerdFont-Medium.ttf"
  "https://github.com/ryanoasis/nerd-fonts/raw/master/patched-fonts/FiraCode/SemiBold/FiraCodeNerdFont-SemiBold.ttf"
  "https://github.com/ryanoasis/nerd-fonts/raw/master/patched-fonts/FiraCode/Bold/FiraCodeNerdFont-Bold.ttf"
)

FALLBACK_URLS=(
  "https://raw.githubusercontent.com/abhisek-dhua/my-linux-setup/main/my-fonts/FiraCodeNerf/FiraCodeNerdFont-Regular.ttf"
  "https://raw.githubusercontent.com/abhisek-dhua/my-linux-setup/main/my-fonts/FiraCodeNerf/FiraCodeNerdFont-Medium.ttf"
  "https://raw.githubusercontent.com/abhisek-dhua/my-linux-setup/main/my-fonts/FiraCodeNerf/FiraCodeNerdFont-SemiBold.ttf"
  "https://raw.githubusercontent.com/abhisek-dhua/my-linux-setup/main/my-fonts/FiraCodeNerf/FiraCodeNerdFont-Bold.ttf"
)

download_fonts() {
  local urls=("$@")
  for url in "${urls[@]}"; do
    filename=$(basename "$url")
    if [[ ! -f "$FONT_DIR/$filename" ]]; then
      wget -q --timeout=30 "$url" -O "$FONT_DIR/$filename" 2>/dev/null || echo "⚠️ Failed to download $filename"
    fi
  done
}

download_fonts "${FONT_URLS[@]}"

if ls "$FONT_DIR"/FiraCodeNerdFont-*.ttf &>/dev/null; then
  echo -e "${GREEN}✅ FiraCode Nerd Font installed${NC}"
else
  echo -e "${YELLOW}⚠️ Primary font source failed, trying fallback...${NC}"
  download_fonts "${FALLBACK_URLS[@]}"
fi

fc-cache -f >/dev/null 2>&1
)
else
  echo -e "${YELLOW}⏭️ Skipping fonts${NC}"
fi

# ═══════════════════════════════════════════════════
# 💻 SECTION 8: Zsh + Oh My Zsh
# ═══════════════════════════════════════════════════
section "💻 Installing Zsh + Oh My Zsh"

echo -e "${BOLD}Install Zsh + Oh My Zsh? (y/n)${NC}"
read -r INSTALL_ZSH

if [[ "$INSTALL_ZSH" =~ ^[Yy]$ ]]; then
sudo apt install -y zsh || true

if [[ "$SWITCH_TO_ZSH" =~ ^[Yy]$ ]]; then
    echo "🔄 Setting zsh as default shell for $SETUP_USER..."
    sudo usermod -s "$(which zsh)" "$SETUP_USER" || chsh -s "$(which zsh)" || true
fi

if [[ ! -d "$USER_HOME/.oh-my-zsh" ]]; then
  echo "⬇️ Installing Oh My Zsh..."
  RUNZSH=no CHSH=no sh -c \
    "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true
  sudo chown -R "$SETUP_USER":"$SETUP_USER" "$USER_HOME/.oh-my-zsh" 2>/dev/null || true
fi

ZSH_CUSTOM="${ZSH_CUSTOM:-$USER_HOME/.oh-my-zsh/custom}"
mkdir -p "$ZSH_CUSTOM/plugins"

install_plugin() {
  local name=$1
  local repo=$2

  if [[ -d "$ZSH_CUSTOM/plugins/$name" ]]; then
    echo -e "${GREEN}✅ $name already installed${NC}"
  else
    echo -e "${CYAN}⬇️ Installing $name...${NC}"
    git clone "$repo" "$ZSH_CUSTOM/plugins/$name" || true
  fi
}

install_plugin "zsh-autosuggestions" "https://github.com/zsh-users/zsh-autosuggestions"
install_plugin "zsh-syntax-highlighting" "https://github.com/zsh-users/zsh-syntax-highlighting.git"
install_plugin "fast-syntax-highlighting" "https://github.com/zdharma-continuum/fast-syntax-highlighting.git"

grep -q "^ZSH_THEME=" "$ZSHRC" \
  && sed -i 's/^ZSH_THEME=.*/ZSH_THEME="agnoster"/' "$ZSHRC" \
  || echo 'ZSH_THEME="agnoster"' >> "$ZSHRC"

grep -q "FiraCode" "$ZSHRC" || echo 'export FIRA_CODE="FiraCode Nerd Font"' >> "$ZSHRC"

grep -q "PROMPT_SEGMENT_USER_BG" "$ZSHRC" || cat >> "$ZSHRC" << 'EOF'

# Agnoster Theme Colors
PROMPT_SEGMENT_USER_BG="blue"
PROMPT_SEGMENT_USER_FG="white"
PROMPT_SEGMENT_DIR_BG="cyan"
PROMPT_SEGMENT_DIR_FG="white"
PROMPT_SEGMENT_GIT_FG="black"
PROMPT_SEGMENT_GIT_CLEAN_BG="green"
PROMPT_SEGMENT_GIT_DIRTY_BG="yellow"
PROMPT_SEGMENT_VENV_FG="white"
PROMPT_SEGMENT_VENV_BG="magenta"
PROMPT_SEGMENT_TIME_BG="default"
EOF

grep -q "^plugins=" "$ZSHRC" \
  && sed -i 's/^plugins=.*/plugins=(git zsh-autosuggestions zsh-syntax-highlighting fast-syntax-highlighting)/' "$ZSHRC" \
  || echo 'plugins=(git zsh-autosuggestions zsh-syntax-highlighting fast-syntax-highlighting)' >> "$ZSHRC"

grep -q HIST_STAMPS "$ZSHRC" || echo 'HIST_STAMPS="yyyy-mm-dd"' >> "$ZSHRC"
else
  echo -e "${YELLOW}⏭️ Skipping Zsh setup${NC}"
fi

# ═══════════════════════════════════════════════════
# ⚡ SECTION 9: NVM + Node
# ═══════════════════════════════════════════════════
section "⚡ Installing NVM + Node"

echo -e "${BOLD}Install NVM + Node? (y/n)${NC}"
read -r INSTALL_NODE

if [[ "$INSTALL_NODE" =~ ^[Yy]$ ]]; then

export NVM_DIR="$USER_HOME/.nvm"

if [[ ! -d "$NVM_DIR" ]]; then
  echo "⬇️ Installing NVM..."
  NVM_VERSION=$(curl -fsSL https://api.github.com/repos/nvm-sh/nvm/releases/latest | grep -o '"tag_name": "[^"]*"' | cut -d'"' -f4)
  if [[ -z "$NVM_VERSION" ]]; then
    echo "⚠️ Could not fetch latest NVM version, falling back to v0.40.1"
    NVM_VERSION="v0.40.1"
  fi
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh | bash || true
fi

grep -q "NVM_DIR" "$ZSHRC" || cat >> "$ZSHRC" << 'EOF'
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
EOF

set +u
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

if command -v nvm >/dev/null 2>&1; then    echo "✅ NVM loaded"
  nvm install --lts || true
  nvm use --lts || true
  # Fix ownership since we installed as root
  sudo chown -R "$SETUP_USER":"$SETUP_USER" "$USER_HOME/.nvm" 2>/dev/null || true

  if command -v npm >/dev/null 2>&1; then
    # Allow postinstall scripts for packages we install
    CURRENT_ALLOW=$(npm config get allow-scripts --location=user 2>/dev/null || echo "")
    PACKAGES="cline,protobufjs,opencode-ai,@kilocode/cli,@angular/cli"
    if [[ -n "$CURRENT_ALLOW" && "$CURRENT_ALLOW" != "null" ]]; then
      # Merge with existing allow-scripts, deduplicate
      MERGED=$(echo "$CURRENT_ALLOW,$PACKAGES" | tr ',' '\n' | sort -u | tr '\n' ',' | sed 's/,$//')
    else
      MERGED="$PACKAGES"
    fi
    npm config set allow-scripts "$MERGED" --location=user 2>/dev/null || true

    echo "🤖 Installing AI CLI tools..."
    npm install -g cline || true
    npm install -g opencode-ai || true
    npm install -g @kilocode/cli || true

    echo "🅰️ Installing Angular CLI..."
    npm install -g @angular/cli || true
  fi
fi
set -u
else
  echo -e "${YELLOW}⏭️ Skipping NVM + Node${NC}"
fi

# ═══════════════════════════════════════════════════
# 🐍 SECTION 10: Python + Pyenv
# ═══════════════════════════════════════════════════
section "🐍 Installing Python + Pyenv"

echo -e "${BOLD}Install Python + Pyenv? (y/n)${NC}"
read -r INSTALL_PYTHON

if [[ "$INSTALL_PYTHON" =~ ^[Yy]$ ]]; then
sudo apt install -y python3 python3-pip python3-venv || true

sudo apt install -y make libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev || true

if [[ ! -d "$USER_HOME/.pyenv" ]]; then
  curl https://pyenv.run | bash || true
  sudo chown -R "$SETUP_USER":"$SETUP_USER" "$USER_HOME/.pyenv" 2>/dev/null || true
fi

grep -q PYENV_ROOT "$ZSHRC" || cat >> "$ZSHRC" << 'EOF'
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF
else
  echo -e "${YELLOW}⏭️ Skipping Python + Pyenv${NC}"
fi

# ═══════════════════════════════════════════════════
# 🖥 SECTION 11: Terminal Config
# ═══════════════════════════════════════════════════
section "🖥 Configuring Terminal"

echo -e "${BOLD}Configure Terminal? (y/n)${NC}"
read -r INSTALL_TERMINAL

if [[ "$INSTALL_TERMINAL" =~ ^[Yy]$ ]]; then

# Detect terminal: Ptyxis (Ubuntu 26+) or GNOME Terminal (Ubuntu 24 and earlier)
if command -v ptyxis-cli &>/dev/null || dpkg -l ptyxis 2>/dev/null | grep -q "^ii"; then
    # Ptyxis terminal (Ubuntu 26.04+)
    echo -e "${CYAN}Detected Ptyxis terminal (Ubuntu 26+)${NC}"

    if command -v gsettings &>/dev/null; then
        PTYXIS_UUID=$(gsettings get org.gnome.Ptyxis default-profile-uuid 2>/dev/null | tr -d "'")

        if [[ -n "$PTYXIS_UUID" ]]; then
            PTYXIS_BASE="org.gnome.Ptyxis.Profile:/org/gnome/Ptyxis/Profiles/${PTYXIS_UUID}/"

            gsettings set "$PTYXIS_BASE" opacity 0.84 2>/dev/null || true
            gsettings set "$PTYXIS_BASE" use-system-font false 2>/dev/null || true
            gsettings set "$PTYXIS_BASE" font 'FiraCode Nerd Font 12' 2>/dev/null || true

            echo "✅ Ptyxis terminal profile configured"
        else
            echo "⚠️ Could not find Ptyxis default profile UUID"
        fi
    else
        echo "⚠️ gsettings not installed, skipping terminal config"
    fi

elif command -v gnome-terminal &>/dev/null; then
    # GNOME Terminal (Ubuntu 24 and earlier)
    echo -e "${CYAN}Detected GNOME Terminal${NC}"

    if command -v gsettings &>/dev/null; then
        PROFILE_PATH=$(gsettings get org.gnome.Terminal.ProfilesList default 2>/dev/null | tr -d "'")

        if [[ -n "$PROFILE_PATH" ]]; then
            BASE="org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/:${PROFILE_PATH}/"

            gsettings set "$BASE" use-system-font false 2>/dev/null || true
            gsettings set "$BASE" use-theme-colors false 2>/dev/null || true
            gsettings set "$BASE" font 'FiraCode Nerd Font 12' 2>/dev/null || true
            gsettings set "$BASE" use-theme-transparency false 2>/dev/null || true
            gsettings set "$BASE" background-color 'rgb(0,0,0)' 2>/dev/null || true
            gsettings set "$BASE" use-transparent-background true 2>/dev/null || true
            gsettings set "$BASE" background-transparency-percent 16 2>/dev/null || true
            gsettings set "$BASE" foreground-color 'rgb(255,255,255)' 2>/dev/null || true

            echo "✅ GNOME Terminal profile configured"
        else
            echo "⚠️ Could not find terminal profile UUID"
        fi
    else
        echo "⚠️ gsettings not installed, skipping terminal config"
    fi
else
    echo "⚠️ No supported terminal found (Ptyxis or GNOME Terminal)"
fi

else
  echo -e "${YELLOW}⏭️ Skipping terminal config${NC}"
fi

# ═══════════════════════════════════════════════════
# 🌐 SECTION 12: Google Chrome
# ═══════════════════════════════════════════════════
section "🌐 Installing Google Chrome"

echo -e "${BOLD}Install Google Chrome? (y/n)${NC}"
read -r INSTALL_CHROME

if [[ "$INSTALL_CHROME" =~ ^[Yy]$ ]]; then

# Fix i386 architecture warning (only if no i386 packages installed)
if dpkg --print-foreign-architectures | grep -q i386 && ! dpkg -l 2>/dev/null | grep -q "^[a-zA-Z].*i386"; then
  sudo dpkg --remove-architecture i386 2>/dev/null || true
fi
sudo rm -f /etc/apt/sources.list.d/google-chrome.list 2>/dev/null || true

wget -qO- https://dl.google.com/linux/linux_signing_key.pub | sudo gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg || true

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | \
  sudo tee /etc/apt/sources.list.d/google-chrome.list >/dev/null

sudo apt update || true
sudo apt install -y google-chrome-stable || true
else
  echo -e "${YELLOW}⏭️ Skipping Google Chrome${NC}"
fi

# ═══════════════════════════════════════════════════
# 🧠 SECTION 13: VS Code
# ═══════════════════════════════════════════════════
section "🧠 Installing VS Code"

echo -e "${BOLD}Install VS Code? (y/n)${NC}"
read -r INSTALL_VSCODE

if [[ "$INSTALL_VSCODE" =~ ^[Yy]$ ]]; then

sudo snap remove code || true

# Remove any existing VSCode sources to prevent duplicates
sudo rm -f /usr/share/keyrings/microsoft.gpg 2>/dev/null || true
sudo rm -f /etc/apt/sources.list.d/vscode.list 2>/dev/null || true
sudo rm -f /etc/apt/sources.list.d/vscode.sources 2>/dev/null || true
sudo rm -f /etc/apt/keyrings/packages.microsoft.gpg 2>/dev/null || true
sudo rm -f /etc/apt/trusted.gpg.d/packages.microsoft.gpg 2>/dev/null || true

# Import Microsoft GPG key (official path per Microsoft docs)
sudo apt install -y wget gpg 2>/dev/null || true
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /usr/share/keyrings/microsoft.gpg || true

# Add repo using DEB822 format (works on Ubuntu 22.04+)
sudo tee /etc/apt/sources.list.d/vscode.sources > /dev/null << 'EOF'
Types: deb
URIs: https://packages.microsoft.com/repos/code
Suites: stable
Components: main
Architectures: amd64,arm64,armhf
Signed-By: /usr/share/keyrings/microsoft.gpg
EOF

sudo apt update || true
sudo apt install -y code || true
else
  echo -e "${YELLOW}⏭️ Skipping VS Code${NC}"
fi

# ═══════════════════════════════════════════════════
# 🔐 SECTION 14: Git Config
# ═══════════════════════════════════════════════════
section "🔐 Git Config"

echo -e "${BOLD}Configure git? (y/n)${NC}"
read -r GIT_CONFIRM

if [[ "$GIT_CONFIRM" =~ ^[Yy]$ ]]; then
  if [[ -z "${GIT_NAME:-}" ]]; then
    read -p "Git username: " GIT_NAME
  fi

  if [[ -z "${GIT_EMAIL:-}" ]]; then
    read -p "Git email: " GIT_EMAIL
  fi

  git config --global user.name "$GIT_NAME"
  git config --global user.email "$GIT_EMAIL"
  git config --global core.pager ""
  git config --global credential.helper store
  echo "✅ Git configured"
else
  echo "⏭️ Skipping git config (set manually with: git config --global user.name/email)"
fi

# ═══════════════════════════════════════════════════
# 📥 SECTION 15: Free Download Manager (FDM)
# ═══════════════════════════════════════════════════
section "📥 Installing Free Download Manager"

echo -e "${BOLD}Install Free Download Manager? (y/n)${NC}"
read -r INSTALL_FDM

if [[ "$INSTALL_FDM" =~ ^[Yy]$ ]]; then

if ! dpkg -l freedownloadmanager &>/dev/null 2>&1; then
  curl -L -o /tmp/freedownloadmanager.deb "https://sourceforge.net/projects/free-download-manager/files/freedownloadmanager.deb/download"
  sudo apt install -y /tmp/freedownloadmanager.deb || true
  rm -f /tmp/freedownloadmanager.deb
else
  echo -e "${GREEN}✅ Free Download Manager already installed${NC}"
fi
else
  echo -e "${YELLOW}⏭️ Skipping Free Download Manager${NC}"
fi

# ═══════════════════════════════════════════════════
# 💬 SECTION 16: Microsoft Teams for Linux
# ═══════════════════════════════════════════════════
section "💬 Installing Microsoft Teams for Linux"

echo -e "${BOLD}Install Microsoft Teams for Linux? (y/n)${NC}"
read -r INSTALL_TEAMS

if [[ "$INSTALL_TEAMS" =~ ^[Yy]$ ]]; then
  sudo mkdir -p /etc/apt/keyrings
  sudo wget -qO /etc/apt/keyrings/teams-for-linux.asc https://repo.teamsforlinux.de/teams-for-linux.asc || true

  sudo tee /etc/apt/sources.list.d/teams-for-linux-packages.sources > /dev/null << 'EOF'
Types: deb
URIs: https://repo.teamsforlinux.de/debian/
Suites: stable
Components: main
Signed-By: /etc/apt/keyrings/teams-for-linux.asc
Architectures: amd64
EOF

  sudo apt update || true
  sudo apt install -y teams-for-linux || true

  # Create autostart entry so Teams launches on boot
  mkdir -p "$USER_HOME/.config/autostart"
  if [[ -f /usr/share/applications/teams-for-linux.desktop ]]; then
    cp /usr/share/applications/teams-for-linux.desktop "$USER_HOME/.config/autostart/"
    chown "$SETUP_USER":"$SETUP_USER" "$USER_HOME/.config/autostart/teams-for-linux.desktop"
    log_ok "Teams autostart entry created"
  else
    log_warn "teams-for-linux.desktop not found, creating manually"
    cat > "$USER_HOME/.config/autostart/teams-for-linux.desktop" << 'DESKTOP'
[Desktop Entry]
Name=Teams for Linux
Exec=/opt/teams-for-linux/teams-for-linux --ozone-platform=x11 %U
Terminal=false
Type=Application
Icon=teams-for-linux
StartupWMClass=teams-for-linux
Comment=Unofficial Microsoft Teams client for Linux using Electron.
MimeType=x-scheme-handler/msteams;
Categories=Chat;Network;Office;
DESKTOP
    chown "$SETUP_USER":"$SETUP_USER" "$USER_HOME/.config/autostart/teams-for-linux.desktop"
  fi
  log_ok "Microsoft Teams installed"
else
  echo -e "${YELLOW}⏭️ Skipping Microsoft Teams${NC}"
fi

# ═══════════════════════════════════════════════════
# 🛠️ SECTION 17: Optional System Utilities
# ═══════════════════════════════════════════════════
section "🛠️ Optional System Utilities"

echo -e "${BOLD}Install optional system utilities? (y/n)${NC}"
read -r INSTALL_UTILS

if [[ "$INSTALL_UTILS" =~ ^[Yy]$ ]]; then
  sudo apt install -y \
    ffmpeg \
    p7zip-full \
    exfatprogs \
    gnome-tweaks \
    trash-cli \
    flatpak \
    htop btop \
    jq \
    tree \
    fzf \
    ripgrep \
    fd-find \
    bat || true

  echo -e "${GREEN}✅ Optional utilities installed${NC}"
else
  echo -e "${YELLOW}⏭️ Skipping optional utilities${NC}"
fi

# ═══════════════════════════════════════════════════
# 🧹 Cleanup
# ═══════════════════════════════════════════════════
sudo apt autoremove -y || true

echo ""
echo "✅ SETUP COMPLETE!"
if [[ "$SHELL" == *"/zsh" ]]; then
    echo "👉 Restart terminal or run: source ~/.zshrc"
else
    echo "👉 Restart terminal or run: source ~/.bashrc"
fi
echo ""
read -p "Press Enter to confirm setup completed successfully..."
