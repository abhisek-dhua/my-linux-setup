#!/bin/bash
# Enhanced script to fix Cursor terminal sandbox AppArmor issue
# This version creates completely permissive profiles for all cursor components

echo "This script will fix the Cursor terminal sandbox AppArmor warning."
echo "It requires sudo permissions to create permissive AppArmor profiles."
echo ""

# Ask for sudo password upfront and cache it
sudo -v

# Keep sudo session alive in the background
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &

echo ""
echo "Step 1: Backing up existing cursor AppArmor profiles..."
timestamp=$(date +%Y%m%d_%H%M%S)
for profile in cursor cursor-sandbox cursor-sandbox-remote; do
    if [ -f "/etc/apparmor.d/$profile" ]; then
        sudo cp "/etc/apparmor.d/$profile" "/etc/apparmor.d/$profile.backup.$timestamp"
        echo "Backed up $profile"
    fi
done

echo ""
echo "Step 2: Creating completely permissive cursor AppArmor profile..."
sudo tee /etc/apparmor.d/cursor > /dev/null << 'EOF'
abi <abi/4.0>,
include <tunables/global>

profile cursor /usr/share/cursor/cursor flags=(unconfined) {
  userns,
  /** rwmlkix,
  capability,
  network,
  mount,
  remount,
  umount,
  pivot_root,
  ptrace,
  signal,
  dbus,
  unix,
  include if exists <local/cursor>
}
EOF
echo "Permissive cursor profile created."

echo ""
echo "Step 3: Creating completely permissive cursor-sandbox AppArmor profile..."
sudo tee /etc/apparmor.d/cursor-sandbox > /dev/null << 'EOF'
abi <abi/4.0>,
include <tunables/global>

profile cursor_sandbox /usr/share/cursor/resources/app/resources/helpers/cursorsandbox flags=(unconfined) {
  userns,
  /** rwmlkix,
  capability,
  network,
  mount,
  remount,
  umount,
  pivot_root,
  ptrace,
  signal,
  dbus,
  unix,
  include if exists <local/cursor-sandbox>
}
EOF
echo "Permissive cursor-sandbox profile created."

echo ""
echo "Step 4: Creating completely permissive cursor-sandbox-remote AppArmor profile..."
sudo tee /etc/apparmor.d/cursor-sandbox-remote > /dev/null << 'EOF'
abi <abi/4.0>,
include <tunables/global>

profile cursor_sandbox_remote /home/*/.cursor-server/bin/*/*/resources/helpers/cursorsandbox flags=(unconfined) {
  userns,
  /** rwmlkix,
  capability,
  network,
  mount,
  remount,
  umount,
  pivot_root,
  ptrace,
  signal,
  dbus,
  unix,
  include if exists <local/cursor-sandbox-remote>
}

profile cursor_sandbox_agent_cli /home/*/.local/share/cursor-agent/versions/*/cursorsandbox flags=(unconfined) {
  userns,
  /** rwmlkix,
  capability,
  network,
  mount,
  remount,
  umount,
  pivot_root,
  ptrace,
  signal,
  dbus,
  unix,
  include if exists <local/cursor-sandbox-remote>
}
EOF
echo "Permissive cursor-sandbox-remote profile created."

echo ""
echo "Step 5: Reloading all AppArmor profiles..."
sudo apparmor_parser -r /etc/apparmor.d/cursor
sudo apparmor_parser -r /etc/apparmor.d/cursor-sandbox
sudo apparmor_parser -r /etc/apparmor.d/cursor-sandbox-remote
echo "All profiles reloaded."

echo ""
echo "Step 6: Closing Cursor..."
sudo pkill -f cursor 2>/dev/null || echo "No Cursor processes found."
sleep 2

echo ""
echo "Step 7: Starting Cursor..."
nohup cursor > /dev/null 2>&1 &

echo ""
echo "✓ Fix completed successfully!"
echo "Cursor has been restarted with completely permissive AppArmor profiles."
echo "The terminal sandbox warning should be completely resolved."