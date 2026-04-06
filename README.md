# Ubuntu Dev Setup Scripts

Automated scripts for setting up a complete Ubuntu development environment with essential tools, configurations, and utilities.

## 📁 Scripts Overview

### 1. `setup.sh` - Ultimate Ubuntu Dev Setup
A comprehensive one-time setup script that installs and configures your entire development environment.

**Features:**
- 🐚 **Zsh + Oh My Zsh** with autosuggestions & syntax highlighting
- ⚡ **NVM + Node.js** + AI CLI tools (Cline, OpenCode) + Angular CLI
- 🐍 **Python + Pyenv** for version management
- 🐳 **Docker** with user permissions
- 🖥️ **GNOME Terminal** configuration (FiraCode Nerd Font, dark theme, transparency)
- 🌐 **Google Chrome** & 🧠 **VS Code** installation
- 🔤 **FiraCode Nerd Font** auto-download
- 🖱️ **Touchpad I2C Fix** for ELAN devices (prevents intermittent disconnects)
- 🔐 **Git configuration** setup
- 🧹 System cleanup
- 📝 **Vim, Neovim, VLC, Tmux** installed by default
- 🛠️ **Optional utilities** (htop, btop, jq, tree, fzf, ripgrep, fd-find, bat, ffmpeg, p7zip-full, gnome-tweaks, trash-cli, flatpak) — prompted during setup

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
