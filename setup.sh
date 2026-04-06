#!/bin/bash
set -euo pipefail

BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

section() {
  echo ""
  echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}  $1${NC}"
  echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════${NC}"
  echo ""
}

echo -e "${BOLD}${GREEN}🚀 Starting Ultimate Ubuntu Dev Setup${NC}"
echo -e "${YELLOW}Sections: System → Essentials → Drivers → Touchpad → Docker → Fonts → Zsh → Node → Python → Terminal → Chrome → VS Code → Git → Antigravity → FDM → Optional Utilities${NC}"

# ═══════════════════════════════════════════════════
# 👤 SECTION 1: User Configuration
# ═══════════════════════════════════════════════════

# ============================================================
# 👤 User Configuration
# ============================================================
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
sudo apt install -y \
  build-essential curl wget git unzip tmux \
  software-properties-common apt-transport-https \
  ca-certificates gnupg lsb-release \
  net-tools dconf-cli fonts-powerline \
  xclip xsel vim neovim vlc || true

# ═══════════════════════════════════════════════════
# 💻 SECTION 4: Drivers
# ═══════════════════════════════════════════════════
section "💻 Installing drivers"
sudo ubuntu-drivers autoinstall || true

# ═══════════════════════════════════════════════════
# 🖱️ SECTION 5: Touchpad Fix (ELAN I2C)
# ═══════════════════════════════════════════════════
section "🖱️ Touchpad I2C power management fix"

sudo apt install -y libinput-tools || true

GRUB_CMDLINE=$(grep '^GRUB_CMDLINE_LINUX_DEFAULT=' /etc/default/grub 2>/dev/null || echo '')
if [[ -n "$GRUB_CMDLINE" ]]; then
  if ! echo "$GRUB_CMDLINE" | grep -q 'i2c_hid.reset_descriptor=1'; then
    sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 i2c_hid.reset_descriptor=1"/' /etc/default/grub
    sudo update-grub || true
    echo -e "${GREEN}✅ I2C HID kernel parameter added (reboot to apply)${NC}"
  else
    echo -e "${GREEN}✅ I2C HID kernel parameter already set${NC}"
  fi
fi

mkdir -p "$HOME/.local/bin"

cat > "$HOME/.local/bin/touchpad-reload" << 'SCRIPT'
#!/bin/bash
echo "🔄 Reloading touchpad drivers..."
sudo modprobe -r i2c_hid_acpi i2c_hid hid_multitouch
sudo modprobe i2c_hid
echo "✅ Touchpad reloaded"
SCRIPT

chmod +x "$HOME/.local/bin/touchpad-reload"
echo -e "${GREEN}✅ 'touchpad-reload' command created for quick fix${NC}"

# ═══════════════════════════════════════════════════
# 🐳 SECTION 6: Docker
# ═══════════════════════════════════════════════════
section "🐳 Installing Docker"
sudo apt install -y docker.io docker-compose || true
sudo systemctl enable docker || true
sudo usermod -aG docker "$SETUP_USER" || true

# ═══════════════════════════════════════════════════
# 🔤 SECTION 7: Fonts
# ═══════════════════════════════════════════════════
section "🔤 Installing FiraCode Nerd Font"
FONT_DIR="$HOME/.local/share/fonts"
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

# ═══════════════════════════════════════════════════
# 💻 SECTION 8: Zsh + Oh My Zsh
# ═══════════════════════════════════════════════════
section "💻 Installing Zsh + Oh My Zsh"
sudo apt install -y zsh || true

if [[ "$SWITCH_TO_ZSH" =~ ^[Yy]$ ]]; then
    echo "🔄 Setting zsh as default shell for $SETUP_USER..."
    sudo usermod -s "$(which zsh)" "$SETUP_USER" || chsh -s "$(which zsh)" || true
fi

if [[ ! -d "$HOME/.oh-my-zsh" ]]; then
  echo "⬇️ Installing Oh My Zsh..."
  RUNZSH=no CHSH=no sh -c \
    "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true
fi

ZSH_CUSTOM="${ZSH_CUSTOM:-$HOME/.oh-my-zsh/custom}"
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

grep -q "^ZSH_THEME=" ~/.zshrc \
  && sed -i 's/^ZSH_THEME=.*/ZSH_THEME="agnoster"/' ~/.zshrc \
  || echo 'ZSH_THEME="agnoster"' >> ~/.zshrc

grep -q "FiraCode" ~/.zshrc || echo 'export FIRA_CODE="FiraCode Nerd Font"' >> ~/.zshrc

grep -q "PROMPT_SEGMENT_USER_BG" ~/.zshrc || cat >> ~/.zshrc << 'EOF'

# Agnoster Theme Colors
PROMPT_SEGMENT_USER_BG="blue"
PROMPT_SEGMENT_USER_FG="white"
PROMPT_SEGMENT_DIR_BG="cyan"
PROMPT_SEGMENT_DIR_FG="white"
PROMPT_SEGMENT_GIT_CLEAN_BG="green"
PROMPT_SEGMENT_GIT_DIRTY_BG="yellow"
PROMPT_SEGMENT_GIT_FG="black"
PROMPT_SEGMENT_VENV_BG="magenta"
PROMPT_SEGMENT_VENV_FG="white"
PROMPT_SEGMENT_TIME_BG="default"
EOF

grep -q "^plugins=" ~/.zshrc \
  && sed -i 's/^plugins=.*/plugins=(git zsh-autosuggestions zsh-syntax-highlighting fast-syntax-highlighting)/' ~/.zshrc \
  || echo 'plugins=(git zsh-autosuggestions zsh-syntax-highlighting fast-syntax-highlighting)' >> ~/.zshrc

grep -q HIST_STAMPS ~/.zshrc || echo 'HIST_STAMPS="yyyy-mm-dd"' >> ~/.zshrc

# ═══════════════════════════════════════════════════
# ⚡ SECTION 9: NVM + Node
# ═══════════════════════════════════════════════════
section "⚡ Installing NVM + Node"

export NVM_DIR="$HOME/.nvm"

if [[ ! -d "$NVM_DIR" ]]; then
  echo "⬇️ Installing NVM..."
  NVM_VERSION=$(curl -fsSL https://api.github.com/repos/nvm-sh/nvm/releases/latest | grep -o '"tag_name": "[^"]*"' | cut -d'"' -f4)
  if [[ -z "$NVM_VERSION" ]]; then
    echo "⚠️ Could not fetch latest NVM version, falling back to v0.40.1"
    NVM_VERSION="v0.40.1"
  fi
  curl -fsSL https://raw.githubusercontent.com/nvm-sh/nvm/${NVM_VERSION}/install.sh | bash || true
fi

grep -q "NVM_DIR" ~/.zshrc || cat >> ~/.zshrc << 'EOF'
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
EOF

set +u
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

if command -v nvm >/dev/null 2>&1; then
  echo "✅ NVM loaded"
  nvm install --lts || true
  nvm use --lts || true

  if command -v npm >/dev/null 2>&1; then
    echo "🤖 Installing AI CLI tools..."
    npm install -g cline || npm install -g https://github.com/cline/cline || true
    npm install -g opencode-ai || true

    echo "🅰️ Installing Angular CLI..."
    npm install -g @angular/cli || true
  fi
fi
set -u

# ═══════════════════════════════════════════════════
# 🐍 SECTION 10: Python + Pyenv
# ═══════════════════════════════════════════════════
section "🐍 Installing Python + Pyenv"
sudo apt install -y python3 python3-pip python3-venv || true

sudo apt install -y make libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev llvm \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev || true

if [[ ! -d "$HOME/.pyenv" ]]; then
  curl https://pyenv.run | bash || true
fi

grep -q PYENV_ROOT ~/.zshrc || cat >> ~/.zshrc << 'EOF'
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF

# ═══════════════════════════════════════════════════
# 🖥 SECTION 11: GNOME Terminal Config
# ═══════════════════════════════════════════════════
section "🖥 Configuring GNOME Terminal"

if command -v gnome-terminal &>/dev/null; then
    if command -v dconf &>/dev/null; then
        PROFILE_PATH=$(dconf list /org/gnome/terminal/legacy/profiles:/ 2>/dev/null | grep -E '^:' | head -1 | tr -d ':')
        
        if [[ -n "$PROFILE_PATH" ]]; then
            BASE="org.gnome.Terminal.Legacy.Profile:/org/gnome/terminal/legacy/profiles:/${PROFILE_PATH}/"
            
            gsettings set "$BASE" font 'FiraCode Nerd Font 12' 2>/dev/null || true
            gsettings set "$BASE" use-system-font false 2>/dev/null || true
            gsettings set "$BASE" use-theme-colors false 2>/dev/null || true
            gsettings set "$BASE" background-color 'rgb(0,0,0)' 2>/dev/null || true
            gsettings set "$BASE" foreground-color 'rgb(255,255,255)' 2>/dev/null || true
            gsettings set "$BASE" use-theme-transparency false 2>/dev/null || true
            gsettings set "$BASE" use-transparent-background true 2>/dev/null || true
            gsettings set "$BASE" background-transparency-percent 16 2>/dev/null || true
            
            echo "✅ Terminal profile configured"
        else
            echo "⚠️ Could not find terminal profile UUID"
        fi
    else
        echo "⚠️ dconf not installed, skipping terminal config"
    fi
fi

# ═══════════════════════════════════════════════════
# 🌐 SECTION 12: Google Chrome
# ═══════════════════════════════════════════════════
section "🌐 Installing Google Chrome"

wget -qO- https://dl.google.com/linux/linux_signing_key.pub | sudo gpg --dearmor -o /usr/share/keyrings/google-chrome.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main" | \
  sudo tee /etc/apt/sources.list.d/google-chrome.list >/dev/null

sudo apt update
sudo apt install -y google-chrome-stable || true

# ═══════════════════════════════════════════════════
# 🧠 SECTION 13: VS Code
# ═══════════════════════════════════════════════════
section "🧠 Installing VS Code"

sudo snap remove code || true

# Remove any existing VSCode sources to prevent duplicates
sudo rm -f /etc/apt/sources.list.d/vscode.list 2>/dev/null || true
sudo rm -f /etc/apt/sources.list.d/vscode.sources 2>/dev/null || true
sudo rm -f /etc/apt/trusted.gpg.d/packages.microsoft.gpg 2>/dev/null || true

# Add repo cleanly
wget -qO- https://packages.microsoft.com/keys/microsoft.asc | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/packages.microsoft.gpg

echo "deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | \
  sudo tee /etc/apt/sources.list.d/vscode.list >/dev/null

sudo apt update
sudo apt install -y code || true

# ═══════════════════════════════════════════════════
# 🔐 SECTION 14: Git Config (Skip with 's')
# ═══════════════════════════════════════════════════
section "🔐 Git Config"
if [[ -z "${GIT_NAME:-}" ]]; then
  read -p "Git username (or 's' to skip): " GIT_NAME
fi

if [[ -z "${GIT_EMAIL:-}" ]]; then
  read -p "Git email (or 's' to skip): " GIT_EMAIL
fi

if [[ "${GIT_NAME:-}" != "s" && "${GIT_EMAIL:-}" != "s" && -n "${GIT_NAME:-}" && -n "${GIT_EMAIL:-}" ]]; then
  git config --global user.name "$GIT_NAME"
  git config --global user.email "$GIT_EMAIL"
  git config --global core.pager ""
  git config --global credential.helper store
  echo "✅ Git configured"
else
  echo "⏭️ Skipping git config (set manually with: git config --global user.name/email)"
fi

# ═══════════════════════════════════════════════════
# 🚗 SECTION 15: Antigravity
# ═══════════════════════════════════════════════════
section "🚗 Installing Antigravity"

sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://us-central1-apt.pkg.dev/doc/repo-signing-key.gpg -o /tmp/antigravity-key-ascii.gpg
gpg --dearmor --yes -o /tmp/antigravity-repo-key.gpg /tmp/antigravity-key-ascii.gpg
sudo cp /tmp/antigravity-repo-key.gpg /etc/apt/keyrings/antigravity-repo-key.gpg
sudo chmod a+r /etc/apt/keyrings/antigravity-repo-key.gpg
rm -f /tmp/antigravity-key-ascii.gpg /tmp/antigravity-repo-key.gpg

echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/antigravity-repo-key.gpg] https://us-central1-apt.pkg.dev/projects/antigravity-auto-updater-dev/ antigravity-debian main" | \
  sudo tee /etc/apt/sources.list.d/antigravity.list >/dev/null

sudo apt update
sudo apt install -y antigravity || true

# ═══════════════════════════════════════════════════
# 📥 SECTION 16: Free Download Manager (FDM)
# ═══════════════════════════════════════════════════
section "📥 Installing Free Download Manager"

if ! dpkg -l freedownloadmanager &>/dev/null 2>&1; then
  curl -L -o /tmp/freedownloadmanager.deb "https://sourceforge.net/projects/free-download-manager/files/freedownloadmanager.deb/download"
  sudo apt install -y /tmp/freedownloadmanager.deb || true
  rm -f /tmp/freedownloadmanager.deb
else
  echo -e "${GREEN}✅ Free Download Manager already installed${NC}"
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
    exfat-fuse exfatprogs \
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
