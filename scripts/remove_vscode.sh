#!/bin/bash

# VS Code Complete Removal Script

commands=()

run() {
  commands+=("$*")
  echo "Running: $*"
  eval "$*"
}

echo "Stopping VS Code processes..."
run "pkill -f code 2>/dev/null || true"

echo "Removing snap version..."
run "sudo snap remove code 2>/dev/null || true"

echo "Removing apt package..."
run "sudo apt remove --purge code -y 2>/dev/null || true"

echo "Removing Microsoft repository..."
run "sudo rm -f /etc/apt/sources.list.d/vscode.list"

echo "Removing GPG key..."
run "sudo rm -f /etc/apt/keyrings/packages.microsoft.gpg"

echo "Removing config directory..."
run "rm -rf ~/.config/Code"

echo "Cleaning up unused packages..."
run "sudo apt autoremove -y"

echo "Updating package lists..."
run "sudo apt update"

echo
echo "Commands executed:"
for cmd in "${commands[@]}"; do
  echo "  $cmd"
done

echo
echo "VS Code completely removed."