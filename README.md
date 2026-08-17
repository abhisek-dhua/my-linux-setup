# Ubuntu Dev Setup Scripts

Automated scripts for setting up a complete Ubuntu development environment with essential tools, configurations, and utilities.

## 📁 Scripts Overview

### 1. `setup.sh` - Ultimate Ubuntu Dev Setup

A comprehensive, idempotent setup script that installs and configures your entire development environment. **Safe to re-run** — it never duplicates configs or breaks existing installs.

**Features (runs in order):**

| #   | Section                  | Mandatory | Description                                                                                                                                                                                               |
| --- | ------------------------ | --------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | 👤 User Configuration    | ✅ Yes    | Username detection + zsh shell prompt                                                                                                                                                                     |
| 2   | 🔄 System Update         | ✅ Yes    | Full `apt update && apt upgrade`                                                                                                                                                                          |
| 3   | 🧩 Essentials            | ❌ No     | build-essential, curl, wget, git, unzip, tmux, software-properties-common, apt-transport-https, ca-certificates, gnupg, lsb-release, net-tools, dconf-cli, fonts-powerline, xclip, xsel, vim, neovim, vlc |
| 4   | 💻 Drivers               | ❌ No     | `ubuntu-drivers autoinstall`                                                                                                                                                                              |
| 5   | 🖱️ Touchpad Fix          | ❌ No     | Generic I2C-HID freeze fix: runtime-PM prevention + resume reset + smart watchdog (delegates to `scripts/fix-touchpad.sh --apply`)                                                                        |
| 6   | 🐳 Docker                | ❌ No     | docker.io + docker-compose (supports both `docker-compose-plugin` and legacy `docker-compose`)                                                                                                            |
| 7   | 🔤 Fonts                 | ❌ No     | FiraCode Nerd Font (auto-download with fallback)                                                                                                                                                          |
| 8   | 💻 Zsh + Oh My Zsh       | ❌ No     | Agnoster theme, zsh-autosuggestions, zsh-syntax-highlighting, fast-syntax-highlighting                                                                                                                    |
| 9   | ⚡ NVM + Node            | ❌ No     | Latest LTS Node + AI Tools (Cline, OpenCode, KiloCode) and Angular CLI                                                                                                                                    |
| 10  | 🐍 Python + Pyenv        | ❌ No     | python3 + pyenv with dev dependencies                                                                                                                                                                     |
| 11  | 🖥 Terminal Config       | ❌ No     | Auto-detects Ptyxis (Ubuntu 26+) or GNOME Terminal (Ubuntu 24), configures font, theme, transparency                                                                                                      |
| 12  | 🌐 Google Chrome         | ❌ No     | Official repo (auto-updates via apt, safe multiarch handling)                                                                                                                                             |
| 13  | 🦊 Mozilla Firefox       | ❌ No     | Mozilla PPA (removes Snap, auto-updates via apt)                                                                                                                                                          |
| 14  | 🧠 VS Code               | ❌ No     | Official repo (DEB822 format, keys in `/etc/apt/keyrings/` for APT 3.1)                                                                                                                                   |
| 15  | 🔐 Git Config            | ❌ No     | Username/email setup + credential helper                                                                                                                                                                  |
| 16  | 📥 Free Download Manager | ❌ No     | FDM (.deb from SourceForge)                                                                                                                                                                               |
| 17  | 💬 Microsoft Teams       | ❌ No     | Community-maintained Teams client with apt repo (auto-updates via apt)                                                                                                                                    |
| 18  | 🛠️ Optional Utilities    | ❌ No     | htop, btop, jq, tree, fzf, ripgrep, fd-find, bat, ffmpeg, p7zip, gnome-tweaks, flatpak, exfatprogs (Ubuntu 26 compatible)                                                                                 |

> **Note:** Sections 1 and 2 are mandatory and run automatically. All other sections (3–18) prompt for confirmation before executing.

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

## 🔁 Re-Running the Script (Safe & Idempotent)

All scripts are designed to be **safe to re-run** — running again days later won't break anything or duplicate configs:

- **apt sources & keys** (Chrome, VS Code, Teams): removed and re-created fresh each run (`rm -f` then `tee`) — never appended, so no duplicate repo entries
- **Shell config (`.zshrc`)**: lines are only added if not already present (`grep -q ... \|\| append`), so no duplicated exports/plugins/themes
- **One-time installs** (Oh My Zsh, plugins, NVM, Pyenv, FDM): skipped automatically if already installed
- **Services** (Docker, touchpad fix): enabled idempotently with `systemctl enable --now`
- **Legacy cleanup**: outdated over-engineered touchpad configs are removed on every run
- **Logs**: each run writes a fresh timestamped log in `/tmp` — previous logs are never overwritten

Re-running simply skips what's already done and re-applies the rest — e.g. useful for installing a section you skipped on the first run.

## 🖱️ Touchpad Fix — How It Works

If your I2C-HID touchpad freezes (stops responding while still "bound" to the driver), it's a known kernel/firmware issue on AMD and Intel I2C controllers, triggered by **suspend/resume** and by **aggressive runtime power management** on the I2C bus. The fix is layered: it prevents the known causes, detects real failures precisely, and recovers within seconds if a freeze still happens.

### Installed layers

| Layer               | File(s)                                                          | What it does                                                                                                                          |
| ------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **1. Prevention**   | (part of fixer + watchdog)                                       | Disables runtime PM for the touchpad, its I2C adapter and the host controller — re-applied continuously, even after rebinds/reboots   |
| **2. Fixer**        | `/usr/local/sbin/touchpad-fixer`                                 | `reset` = power fix + unbind → rebind; `probe` = full i2c-hid driver reload (escalation); `verify` = health check                     |
| **3. Resume**       | `/etc/systemd/system/touchpad-fixer-resume.service`              | Runs the reset automatically after suspend/resume (the classic freeze trigger)                                                       |
| **4. Watchdog**     | `/usr/local/sbin/touchpad-watchdog` + `touchpad-watchdog.service` | Resets ONLY on real failure signals: kernel I2C errors, device missing/unbound, or the touchpad going silent *after working* — confidence-scored so a freeze is caught even if you're only touching the touchpad (~60s while typing, ~90s pad-only). Every recovery is verified; repeated failures escalate to a driver reload |

Works with any ACPI I2C-HID touchpad (ELAN, Synaptics, …) on any systemd-based distro. A merely idle touchpad is never touched.

✅ **Auto-upgrade:** Installing also removes over-engineered leftovers from older versions (udev autosuspend rule, power-management services, systemd-sleep hook).

**Manual recovery (when frozen):**

```bash
sudo /usr/local/sbin/touchpad-fixer reset
```

### `scripts/fix-touchpad.sh` — Touchpad Fix Manager

Interactive menu to manage the fix:

- **1) Apply/upgrade fixes** — installs everything and cleans up legacy leftovers
- **2) Quick restart touchpad** — run a reset when the touchpad freezes
- **3) Check status** — device binding + input node + runtime PM + service state + recent watchdog log
- **4) Uninstall all fixes**

Non-interactive flags (also used by `setup.sh`): `--apply`, `--reset`, `--uninstall`, `--status`.

```bash
chmod +x scripts/fix-touchpad.sh
sudo bash scripts/fix-touchpad.sh            # interactive menu
sudo bash scripts/fix-touchpad.sh --apply    # install/upgrade non-interactively
bash scripts/fix-touchpad.sh --status        # check state (no root needed)
```

**Standalone `setup.sh`:** if you downloaded only `setup.sh` (no repo clone), the touchpad section automatically downloads the latest `scripts/fix-touchpad.sh` from this repo's `main` branch, so it always applies the current version. A local copy always takes precedence.

> **Honest limits:** a userspace fix cannot *guarantee* a hardware/firmware fault never occurs — but this stack prevents the known causes, never touches a healthy touchpad, and restores a genuinely frozen one within seconds.

## 🛠️ Customization

- Edit `setup.sh` to add/remove packages or change configurations
- Place your existing SSH keys in `ssh.zip` to restore them automatically
