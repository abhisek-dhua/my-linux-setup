# Ubuntu Dev Setup Scripts

Automated scripts for setting up a complete Ubuntu development environment with essential tools, configurations, and utilities.

## 📁 Scripts Overview

### 1. `setup.sh` - Ultimate Ubuntu Dev Setup
A comprehensive one-time setup script that installs and configures your entire development environment.

**Features (runs in order):**

| # | Section | Description |
|---|---------|-------------|
| 1 | 👤 User Configuration | Username detection + zsh shell prompt |
| 2 | 🔄 System Update | Full `apt update && apt upgrade` |
| 3 | 🧩 Essentials | build-essential, curl, wget, git, vim, neovim, vlc, tmux |
| 4 | 💻 Drivers | `ubuntu-drivers autoinstall` |
| 5 | 🖱️ Touchpad I2C Fix | ELAN device disconnect fix + `touchpad-reload` command |
| 6 | 🐳 Docker | docker.io + user permissions |
| 7 | 🔤 Fonts | FiraCode Nerd Font (auto-download with fallback) |
| 8 | 💻 Zsh + Oh My Zsh | Agnoster theme, autosuggestions, syntax highlighting |
| 9 | ⚡ NVM + Node | Latest LTS Node + Cline, OpenCode, Angular CLI |
| 10 | 🐍 Python + Pyenv | python3 + pyenv with dev dependencies |
| 11 | 🖥 GNOME Terminal | FiraCode font, dark theme, transparency |
| 12 | 🌐 Google Chrome | Official repo (auto-updates via apt) |
| 13 | 🧠 VS Code | Official repo (auto-updates via apt) |
| 14 | 🔐 Git Config | Username/email setup *(type `s` to skip)* |
| 15 | 🚗 Antigravity | Google Antigravity app with apt repo |
| 16 | 📥 Free Download Manager | FDM with apt repo (auto-configured by .deb) |
| 17 | 🛠️ Optional Utilities | *(prompted)* htop, btop, jq, tree, fzf, ripgrep, fd-find, bat, ffmpeg, p7zip, gnome-tweaks, flatpak |

**Usage:**
```bash
chmod +x setup.sh
./setup.sh
```

### 2. `ssh-key-manager.sh` - SSH Key Manager
Interactive menu-driven tool for managing SSH keys.

**Features:**
- 🔑 **Add SSH key** → restore from `ssh.zip` or create new
- 💾 **Backup** → save current keys to `ssh.zip`
- 👁️ **View public key** → display all public keys
- 🗑️ **Remove keys** → safely delete with automatic backup
- 🤖 **Auto-add to ssh-agent**

**Usage:**
```bash
chmod +x ssh-key-manager.sh
./ssh-key-manager.sh
```

## 🚀 Quick Start

1. Clone this repository
2. Run the main setup script:
   ```bash
   ./setup.sh
   ```
3. Manage SSH keys:
   ```bash
   ./ssh-key-manager.sh
   ```

## 📋 Requirements
- Ubuntu 20.04+ (or Debian-based distro)
- Internet connection
- Sudo privileges

## 🛠️ Customization
- Edit `setup.sh` to add/remove packages or change configurations
- Place your existing SSH keys in `ssh.zip` to restore them automatically
