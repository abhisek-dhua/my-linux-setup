# Ubuntu Dev Setup Scripts

Automated scripts for setting up a complete Ubuntu development environment with essential tools, configurations, and utilities.

## 📁 Scripts Overview

### 1. `setup.sh` - Ultimate Ubuntu Dev Setup

A comprehensive one-time setup script that installs and configures your entire development environment.

**Features (runs in order):**

| #   | Section                  | Mandatory | Description                                                                                                                                                                                               |
| --- | ------------------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 👤 User Configuration    | ✅ Yes    | Username detection + zsh shell prompt                                                                                                                                                                     |
| 2   | 🔄 System Update         | ✅ Yes    | Full `apt update && apt upgrade`                                                                                                                                                                          |
| 3   | 🧩 Essentials            | ❌ No     | build-essential, curl, wget, git, unzip, tmux, software-properties-common, apt-transport-https, ca-certificates, gnupg, lsb-release, net-tools, dconf-cli, fonts-powerline, xclip, xsel, vim, neovim, vlc |
| 4   | 💻 Drivers               | ❌ No     | `ubuntu-drivers autoinstall`                                                                                                                                                                              |
| 5   | 🖱️ Touchpad Fix          | ❌ No     | ELAN1300 targeted idle/sleep fix, udev rule + lightweight systemd-sleep hook, auto cleanup old config                                                                                                     |
| 6   | 🐳 Docker                | ❌ No     | docker.io + docker-compose (supports both `docker-compose-plugin` and legacy `docker-compose`)                                                                                                            |
| 7   | 🔤 Fonts                 | ❌ No     | FiraCode Nerd Font (auto-download with fallback)                                                                                                                                                          |
| 8   | 💻 Zsh + Oh My Zsh       | ❌ No     | Agnoster theme, zsh-autosuggestions, zsh-syntax-highlighting, fast-syntax-highlighting                                                                                                                    |
| 9   | ⚡ NVM + Node            | ❌ No     | Latest LTS Node + AI Tools (Cline, OpenCode, KiloCode) and Angular CLI                                                                                                                                    |
| 10  | 🐍 Python + Pyenv        | ❌ No     | python3 + pyenv with dev dependencies                                                                                                                                                                     |
| 11  | 🖥 Terminal Config       | ❌ No     | Auto-detects Ptyxis (Ubuntu 26+) or GNOME Terminal (Ubuntu 24), configures font, theme, transparency                                                                                                      |
| 12  | 🌐 Google Chrome         | ❌ No     | Official repo (auto-updates via apt, safe multiarch handling)                                                                                                                                             |
| 13  | 🧠 VS Code               | ❌ No     | Official repo (DEB822 format, keys in `/etc/apt/keyrings/` for APT 3.1)                                                                                                                                   |
| 14  | 🔐 Git Config            | ❌ No     | Username/email setup + credential helper                                                                                                                                                                  |
| 15  | 📥 Free Download Manager | ❌ No     | FDM (.deb from SourceForge)                                                                                                                                                                               |
| 16  | 💬 Microsoft Teams       | ❌ No     | Community-maintained Teams client with apt repo (auto-updates via apt)                                                                                                                                    |
| 17  | 🛠️ Optional Utilities    | ❌ No     | htop, btop, jq, tree, fzf, ripgrep, fd-find, bat, ffmpeg, p7zip, gnome-tweaks, flatpak, exfatprogs (Ubuntu 26 compatible)                                                                                 |

> **Note:** Sections 1 and 2 are mandatory and run automatically. All other sections (3–16) prompt for confirmation before executing.

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

- Ubuntu 20.04+ (tested on Ubuntu 24.04 and 26.04, also works on Debian-based distros)
- Internet connection
- Sudo privileges

## 🖱️ Touchpad Idle Fix — How It Works

If your touchpad stops responding after the screen locks or the laptop sits idle, this is caused by Linux's **runtime autosuspend** powering off the I2C touchpad device.

### 🎯 Udev Rule Modes

The script generates a rules file with **3 selectable modes**. Edit `/etc/udev/rules.d/99-touchpad-fix.rules` and uncomment **one only**:

| Mode               | Rule                      | Use case                                                  |
| ------------------ | ------------------------- | --------------------------------------------------------- |
| ✅ **Recommended** | `ATTR{name}=="ELAN1300*"` | Default for modern Lenovo/Asus laptops. Zero side effects |
| 🟡 **Alternative** | `DRIVER=="i2c_hid"`       | All I2C HID input devices, for other touchpad models      |
| 🔴 **Broad**       | `KERNELS=="i2c-*"`        | Last resort, matches everything on I2C bus                |

### 🛠️ Applied Layers:

| Layer                         | File                                        | What it does                                                               |
| ----------------------------- | ------------------------------------------- | -------------------------------------------------------------------------- |
| **1. Kernel param**           | `/etc/default/grub`                         | Adds `i2c_hid.reset_descriptor=1` to fix I2C HID descriptor reset failures |
| **2. Udev rule**              | `/etc/udev/rules.d/99-touchpad-fix.rules`   | Disables autosuspend (choose your match mode above)                        |
| **3. Lightweight sleep hook** | `/lib/systemd/system-sleep/touchpad-fix.sh` | Minimal module reload after suspend resume                                 |

✅ **Auto cleanup:** Script automatically removes old conflicting touchpad configuration files from previous versions

> **Note:** A reboot is required for the GRUB kernel parameter to take effect.

## 🛠️ Customization

- Edit `setup.sh` to add/remove packages or change configurations
- Place your existing SSH keys in `ssh.zip` to restore them automatically
