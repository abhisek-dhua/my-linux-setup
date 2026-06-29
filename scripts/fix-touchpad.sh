#!/bin/bash
set -euo pipefail

BOLD='\033[1m'
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BOLD}${CYAN}"
echo "╔═══════════════════════════════════════════╗"
echo "║        🖱️  Touchpad Fix Script             ║"
echo "╚═══════════════════════════════════════════╝"
echo -e "${NC}"

# ── Prompt sudo password upfront ─────────────────────────────────────────────
echo -e "${BOLD}🔐 This script requires sudo. Please enter your password:${NC}"
sudo -v

# Keep sudo session alive for the duration of the script
while true; do sudo -n true; sleep 50; kill -0 "$$" || exit; done 2>/dev/null &
SUDO_KEEP_ALIVE_PID=$!
trap 'kill "$SUDO_KEEP_ALIVE_PID" 2>/dev/null' EXIT

echo ""

# ── Step 1: Install libinput-tools ───────────────────────────────────────────
echo -e "${BOLD}[1/5] Installing libinput-tools...${NC}"
sudo apt install -y libinput-tools > /dev/null 2>&1 && \
  echo -e "${GREEN}✅ libinput-tools installed${NC}" || \
  echo -e "${YELLOW}⚠️  libinput-tools install failed (continuing)${NC}"

# ── Step 2: Kernel parameter ─────────────────────────────────────────────────
echo -e "${BOLD}[2/5] Checking I2C HID kernel parameter...${NC}"
GRUB_CMDLINE=$(grep '^GRUB_CMDLINE_LINUX_DEFAULT=' /etc/default/grub 2>/dev/null || echo '')
if [[ -n "$GRUB_CMDLINE" ]]; then
  if ! echo "$GRUB_CMDLINE" | grep -q 'i2c_hid.reset_descriptor=1'; then
    sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 i2c_hid.reset_descriptor=1"/' /etc/default/grub
    sudo update-grub > /dev/null 2>&1 || true
    echo -e "${GREEN}✅ Kernel parameter added (reboot to apply)${NC}"
  else
    echo -e "${GREEN}✅ Kernel parameter already set${NC}"
  fi
fi

# ── Step 3: udev rule ────────────────────────────────────────────────────────
echo -e "${BOLD}[3/5] Applying udev rule (disable autosuspend)...${NC}"
cat << 'EOF' | sudo tee /etc/udev/rules.d/99-touchpad-no-autosuspend.rules > /dev/null
# Disable runtime autosuspend for I2C HID input devices (touchpad idle fix)
ACTION=="add", SUBSYSTEM=="i2c", DRIVER=="i2c_hid", ATTR{power/control}="on"
ACTION=="add", SUBSYSTEM=="i2c", DRIVER=="i2c_hid_acpi", ATTR{power/control}="on"
ACTION=="add", SUBSYSTEM=="hid", KERNELS=="i2c-*", ATTR{power/control}="on"
EOF
sudo udevadm control --reload-rules 2>/dev/null || true
sudo udevadm trigger 2>/dev/null || true
echo -e "${GREEN}✅ udev rule applied${NC}"

# ── Step 4: systemd-sleep hook ───────────────────────────────────────────────
echo -e "${BOLD}[4/5] Installing systemd-sleep hook (resume fix)...${NC}"
cat << 'EOF' | sudo tee /lib/systemd/system-sleep/touchpad-resume.sh > /dev/null
#!/bin/bash
# Reload I2C HID touchpad driver after suspend/hibernate resume
if [[ "$1" == "post" ]]; then
  sleep 1
  modprobe -r i2c_hid_acpi 2>/dev/null || true
  modprobe -r i2c_hid      2>/dev/null || true
  modprobe    i2c_hid      2>/dev/null || true
  modprobe    i2c_hid_acpi 2>/dev/null || true
  for dev in /sys/bus/i2c/devices/*/power/control; do
    echo on > "$dev" 2>/dev/null || true
  done
fi
EOF
sudo chmod +x /lib/systemd/system-sleep/touchpad-resume.sh
echo -e "${GREEN}✅ Sleep hook installed${NC}"

# ── Step 5: systemd service + immediate driver reload ────────────────────────
echo -e "${BOLD}[5/5] Enabling touchpad power service & reloading drivers now...${NC}"
cat << 'EOF' | sudo tee /etc/systemd/system/touchpad-persist.service > /dev/null
[Unit]
Description=Keep touchpad powered on (disable autosuspend)
After=multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/bin/bash -c 'for dev in /sys/bus/i2c/devices/*/power/control; do echo on > "$dev" 2>/dev/null || true; done'

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload 2>/dev/null || true
sudo systemctl enable --now touchpad-persist.service 2>/dev/null || true

# Reload drivers immediately so touchpad works right now
sudo modprobe -r i2c_hid_acpi 2>/dev/null || true
sudo modprobe -r i2c_hid      2>/dev/null || true
sudo modprobe    i2c_hid      2>/dev/null || true
sudo modprobe    i2c_hid_acpi 2>/dev/null || true
for dev in /sys/bus/i2c/devices/*/power/control; do
  echo on | sudo tee "$dev" > /dev/null 2>&1 || true
done

echo -e "${GREEN}✅ Drivers reloaded${NC}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}🎉 All touchpad fixes applied!${NC}"
echo -e "${YELLOW}⚠️  A reboot is recommended for the kernel parameter to take full effect.${NC}"
echo ""
