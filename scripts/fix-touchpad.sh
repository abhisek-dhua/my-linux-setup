#!/bin/bash

# Touchpad Fix Manager v2
#
# Generic fix for I2C-HID touchpads (ELAN, Synaptics, ... — any ACPI device
# with modalias PNP0C50/MSFT0001) that freeze: bound to the driver but
# silently unresponsive. Works on any Linux distro with systemd.
#
# Layers installed:
#   1. Prevention  — runtime power management disabled for the touchpad,
#                    its I2C adapter and the host controller (a documented
#                    freeze cause on AMD/Intel laptops). Re-applied constantly.
#   2. Fixer       — /usr/local/sbin/touchpad-fixer
#                    {status|reset|power|probe|verify}
#   3. Resume      — touchpad-fixer-resume.service: power fix + reset after
#                    suspend/hibernate (the classic freeze trigger).
#   4. Watchdog    — /usr/local/sbin/touchpad-watchdog + service. Resets the
#                    touchpad ONLY on demonstrable failure signals:
#                      a. kernel-reported I2C errors for this device
#                      b. touchpad went silent AFTER working, while the user
#                         is actively using other inputs (edge-triggered,
#                         never fires on a merely idle touchpad)
#                      c. device missing / unbound from its driver
#                      d. fallback: device present but silent since boot
#                    Every recovery is verified; repeated failures escalate
#                    to a full i2c-hid driver reload.
#
# Usage:
#   sudo bash scripts/fix-touchpad.sh              # interactive menu
#   sudo bash scripts/fix-touchpad.sh --apply      # install/upgrade (non-interactive)
#   sudo bash scripts/fix-touchpad.sh --reset      # reset touchpad now
#   sudo bash scripts/fix-touchpad.sh --uninstall  # remove everything
#   bash  scripts/fix-touchpad.sh --status         # show state (no root needed)

# ── Preflight ──────────────────────────────────────────────────────────────────
if ! command -v sudo &>/dev/null; then
  echo "❌ sudo is required but not found."
  exit 1
fi

# ── Colors ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Print helpers ──────────────────────────────────────────────────────────────
print_header() {
  echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
  echo -e "${BLUE}   🖱️  Touchpad Fix Manager v2${NC}"
  echo -e "${BLUE}═══════════════════════════════════════════════${NC}"
  echo ""
}

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }
print_ok() { echo -e "${GREEN}[OK]${NC} $1"; }

# ── Embedded fixer script (written to /usr/local/sbin/touchpad-fixer) ─────────
fixer_script() {
  cat << 'EOF'
#!/bin/bash
# touchpad-fixer v2 — recovery + prevention for I2C-HID touchpads.
# Actions: status | reset | power | probe | verify
set -uo pipefail
ACTION="${1:-status}"
LOG_TAG="touchpad-fixer"
log() { logger -t "$LOG_TAG" "$*"; }

find_devices() {
  local sysfs
  for sysfs in /sys/bus/i2c/devices/*-*/; do
    [[ -e "$sysfs/modalias" ]] || continue
    grep -q "PNP0C50\|MSFT0001" "$sysfs/modalias" 2>/dev/null && echo "${sysfs%/}"
  done
  return 0
}

driver_of() {
  local sysfs="$1"
  [[ -L "$sysfs/driver" ]] || { echo ""; return 0; }
  basename "$(readlink -f "$sysfs/driver")"
}

event_node_of() { # prints /dev/input/eventX for a device, empty if none
  local sysfs="$1" ev
  # Trailing slash matters: the /sys/bus/i2c path is a symlink and find
  # does not follow symlink start points without it.
  ev=$(find "$sysfs/" -type d -name "event*" 2>/dev/null | head -1)
  [[ -n "$ev" ]] && echo "/dev/input/$(basename "$ev")"
  return 0
}

# Bind/unbind must use the DEVICE NAME (e.g. ELAN1300:00), not the driver name.
bind_device() {
  local sysfs="$1" drv="$2"
  [[ -L "$sysfs/driver" ]] && return 0
  modprobe "$drv" 2>/dev/null || true
  echo "$(basename "$sysfs")" > "/sys/bus/i2c/drivers/$drv/bind" 2>/dev/null || true
}

unbind_device() {
  local sysfs="$1" drv
  [[ -L "$sysfs/driver" ]] || return 0
  drv=$(driver_of "$sysfs")
  [[ -z "$drv" ]] && return 0
  echo "$(basename "$sysfs")" > "/sys/bus/i2c/drivers/$drv/unbind" 2>/dev/null || true
}

# Prevention: aggressive runtime PM on the touchpad / its I2C adapter / the
# host controller is a documented freeze cause (AMD & Intel). Keeping them
# in "on" costs a trickle of battery but stops the bus from being suspended
# under the touchpad. Walks up the sysfs tree so it works for both platform
# (AMDI0010) and PCI-based (Intel LPSS) controllers.
power_fix_one() {
  local sysfs="$1" p i
  [[ -w "$sysfs/power/control" ]] && echo on > "$sysfs/power/control" 2>/dev/null || true
  p=$(readlink -f "$sysfs")
  # Walk up: touchpad -> I2C adapter (i2c-N) -> host controller (AMDI0010:xx
  # on AMD, i2c_designware.N or a PCI device on Intel).
  for i in 1 2; do
    p=$(dirname "$p")
    [[ -w "$p/power/control" ]] && echo on > "$p/power/control" 2>/dev/null || true
  done
  return 0
}

power_fix_all() {
  local s
  for s in $(find_devices); do power_fix_one "$s"; done
  return 0
}

reset_all() {
  local found=0 sysfs
  power_fix_all
  for sysfs in $(find_devices); do
    found=1
    unbind_device "$sysfs"
    sleep 0.5
    bind_device "$sysfs" "i2c_hid_acpi" || bind_device "$sysfs" "i2c_hid" || true
    sleep 0.5
  done
  power_fix_all
  [[ "$found" -eq 0 ]] && return 1
  log "reset complete ($found device(s))"
  return 0
}

# Last-resort escalation: reload the whole i2c-hid stack.
probe_all() {
  local sysfs
  for sysfs in $(find_devices); do unbind_device "$sysfs"; done
  sleep 0.5
  modprobe -r i2c_hid_acpi 2>/dev/null || true
  modprobe -r i2c_hid 2>/dev/null || true
  sleep 1
  modprobe i2c_hid 2>/dev/null || true
  modprobe i2c_hid_acpi 2>/dev/null || true
  sleep 1
  for sysfs in $(find_devices); do
    bind_device "$sysfs" "i2c_hid_acpi" || bind_device "$sysfs" "i2c_hid" || true
  done
  power_fix_all
  log "i2c-hid driver stack reloaded"
  return 0
}

# Verify every detected touchpad is bound AND exposes an input event node.
verify_all() {
  local s n=0 ok=0
  while IFS= read -r s; do
    [[ -z "$s" ]] && continue
    n=$((n + 1))
    [[ -n "$(driver_of "$s")" ]] || continue
    [[ -n "$(event_node_of "$s")" ]] || continue
    ok=$((ok + 1))
  done < <(find_devices)
  (( n > 0 && ok == n ))
}

case "$ACTION" in
  reset)
    reset_all && echo "✅ Touchpad reset" || echo "⚠️ No I2C-HID touchpad found"
    ;;
  power)
    power_fix_all
    echo "Runtime PM state of the touchpad I2C chain (want: on):"
    for sysfs in $(find_devices); do
      p=$(readlink -f "$sysfs")
      for i in 0 1 2; do
        [[ -r "$p/power/control" ]] && printf "  %-22s %s\n" "$(basename "$p")" "$(cat "$p/power/control")"
        p=$(dirname "$p")
      done
    done
    exit 0
    ;;
  probe)
    probe_all && echo "✅ i2c-hid driver stack reloaded"
    ;;
  verify)
    verify_all && echo "✅ Touchpad bound with input node" || { echo "❌ Touchpad not functional"; exit 1; }
    ;;
  status)
    echo "=== I2C-HID device state ==="
    local_count=0
    for sysfs in $(find_devices); do
      local_count=$((local_count + 1))
      drv="NOT BOUND"
      [[ -L "$sysfs/driver" ]] && drv=$(basename "$(readlink -f "$sysfs/driver")")
      ev=$(event_node_of "$sysfs")
      [[ -z "$ev" ]] && ev="(no input node)"
      pm="n/a"
      [[ -r "$sysfs/power/control" ]] && pm=$(cat "$sysfs/power/control")
      printf "  %-20s bind=%-14s input=%-14s runtime-pm=%s\n" "$(basename "$sysfs")" "$drv" "$(basename "$ev")" "$pm"
    done
    [[ "$local_count" -eq 0 ]] && echo "  (no I2C-HID touchpad found)"
    exit 0
    ;;
  *)
    echo "Usage: $0 {status|reset|power|probe|verify}"
    exit 1
    ;;
esac
EOF
}

# ── Embedded watchdog daemon (written to /usr/local/sbin/touchpad-watchdog) ───
watchdog_script() {
  cat << 'EOF'
#!/bin/bash
# touchpad-watchdog v2.1 — recovers a frozen I2C-HID touchpad from real
# failure signals. The silence heuristic is confidence-scored so it also
# catches a freeze while the user is touching ONLY the touchpad:
#   +2  touchpad silent AND other input active (user is clearly working)
#   +1  touchpad silent, no other input, but desktop session not idle
# The score clears only when the touchpad itself emits events again.
set -u
LOG_TAG="touchpad-watchdog"
FIXER=/usr/local/sbin/touchpad-fixer

# Tunables
CYCLE=10             # loop pacing (event sampling may extend a cycle)
SAMPLE_SECS=25       # touchpad silence that counts as one suspect cycle
SUSPECT_THRESHOLD=3  # weighted score that triggers a heuristic reset
HEUR_COOLDOWN=240    # min seconds between heuristic resets
KMSG_COOLDOWN=45     # min seconds between kernel-error resets
MISSING_COOLDOWN=60  # min seconds between missing-device resets
DEAD_COOLDOWN=600    # fallback reset interval for a never-active touchpad
ESCALATE_AFTER=2     # failed recoveries before full driver re-probe

log() { logger -t "$LOG_TAG" "$*"; }
now() { date +%s; }

find_devices() {
  local sysfs
  for sysfs in /sys/bus/i2c/devices/*-*/; do
    [[ -e "$sysfs/modalias" ]] || continue
    grep -q "PNP0C50\|MSFT0001" "$sysfs/modalias" 2>/dev/null && echo "${sysfs%/}"
  done
  return 0
}

event_node_of() {
  local sysfs="$1" ev
  # Trailing slash: the /sys/bus/i2c path is a symlink; find won't follow
  # a symlink start point without it.
  ev=$(find "$sysfs/" -type d -name "event*" 2>/dev/null | head -1)
  [[ -n "$ev" ]] && echo "/dev/input/$(basename "$ev")"
  return 0
}

first_event_node() {
  local s node
  while IFS= read -r s; do
    [[ -z "$s" ]] && continue
    node=$(event_node_of "$s")
    [[ -n "$node" ]] && { echo "$node"; return 0; }
  done < <(find_devices)
  return 1
}

sample_events() { # true if the touchpad emits anything within SAMPLE_SECS
  timeout "$SAMPLE_SECS" dd if="$1" bs=24 count=1 2>/dev/null | grep -q .
}

other_activity() { # true if any other input device is in use right now
  local tp="$1" ev
  for ev in /dev/input/event*; do
    [[ -c "$ev" && "$ev" != "$tp" ]] || continue
    timeout 2 dd if="$ev" bs=24 count=1 2>/dev/null | grep -q . && return 0
  done
  return 1
}

session_active() { # false only if every login session reports idle
  local sess hint found=0
  while IFS=" " read -r sess _; do
    [[ "$sess" =~ ^[0-9]+$ ]] || continue
    found=1
    hint=$(loginctl show-session "$sess" -p IdleHint --value 2>/dev/null)
    [[ "$hint" == "no" ]] && return 0
  done < <(loginctl list-sessions --no-legend 2>/dev/null)
  # No session info (headless/TTY): assume active so detection stays armed.
  [[ $found -eq 0 ]] && return 0
  return 1
}

# Positive freeze signal: the kernel logged I2C/HID errors for our device
# (or the i2c_hid driver) since the given epoch timestamp.
kernel_errors() {
  local since="$1" devs="" s pat
  for s in $(find_devices); do devs="${devs:+$devs|}$(basename "$s")"; done
  pat="i2c[-_]hid[^:]*: *(timeout|timed out|failed|error|aborted|nack)"
  [[ -n "$devs" ]] && pat="$pat|($devs).*(timeout|timed out|failed|error|aborted|nack)"
  journalctl -kq --no-pager --since "@$since" 2>/dev/null | grep -Eqi "$pat"
}

# Reset, verify the device actually came back, escalate if it keeps failing.
recover() {
  local reason="$1"
  log "resetting touchpad ($reason)"
  "$FIXER" reset >/dev/null 2>&1 || true
  sleep 3
  if "$FIXER" verify >/dev/null 2>&1; then
    FAILED_RECOVERIES=0
    log "recovered ($reason)"
  else
    FAILED_RECOVERIES=$((FAILED_RECOVERIES + 1))
    log "recovery attempt $FAILED_RECOVERIES failed ($reason)"
    if (( FAILED_RECOVERIES >= ESCALATE_AFTER )); then
      log "escalating: reloading i2c-hid driver stack"
      "$FIXER" probe >/dev/null 2>&1 || true
      sleep 3
      if "$FIXER" verify >/dev/null 2>&1; then
        FAILED_RECOVERIES=0
        log "recovered via driver reload ($reason)"
      else
        log "CRITICAL: touchpad did not recover even after driver reload ($reason)"
      fi
    fi
  fi
  TP_WAS_ACTIVE=0
  SUSPECTS=0
}

# ── State ──
TP_WAS_ACTIVE=0      # touchpad emitted events in the previous cycle
SEEN_ACTIVE=0        # touchpad emitted events at least once this boot
SUSPECTS=0           # weighted silence score
FAILED_RECOVERIES=0
LAST_HEUR=0
LAST_KMSG=0
LAST_MISSING=0
LAST_DEAD=$(now) # defer the silent-since-boot fallback, don't fire at startup
LAST_KMSG_CHECK=$(now)

while true; do
  t_start=$(now)

  # Layer 1: prevention — keep runtime PM off even after rebinds/reboots.
  "$FIXER" power >/dev/null 2>&1 || true

  node=$(first_event_node || true)

  if [[ -z "$node" ]]; then
    # Signal c: device missing or unbound — unambiguous, always recoverable.
    t=$(now)
    if (( t - LAST_MISSING >= MISSING_COOLDOWN )); then
      LAST_MISSING=$t
      recover "device missing or unbound"
    fi
    TP_WAS_ACTIVE=0
  elif sample_events "$node"; then
    TP_WAS_ACTIVE=1
    SEEN_ACTIVE=1
    SUSPECTS=0
  else
    t=$(now)
    # Signal b: went silent AFTER working. Scored by confidence so a
    # pad-only user (touching a dead touchpad, typing nothing) is still
    # caught; the score only clears when the touchpad emits events again.
    if (( TP_WAS_ACTIVE == 1 )); then
      if other_activity "$node"; then
        SUSPECTS=$((SUSPECTS + 2))
        log "silence score $SUSPECTS/$SUSPECT_THRESHOLD (other input active)"
      elif session_active; then
        SUSPECTS=$((SUSPECTS + 1))
        log "silence score $SUSPECTS/$SUSPECT_THRESHOLD (no other input, session active)"
      fi
      if (( SUSPECTS >= SUSPECT_THRESHOLD && t - LAST_HEUR >= HEUR_COOLDOWN )); then
        LAST_HEUR=$t
        recover "silent after active use"
      fi
    fi
    # Signal d: device present but has never emitted anything this boot
    # while the user is active — periodic low-frequency reset fallback.
    if (( SEEN_ACTIVE == 0 )) && other_activity "$node" && (( t - LAST_DEAD >= DEAD_COOLDOWN )); then
      LAST_DEAD=$t
      recover "no events since boot"
    fi
  fi

  # Signal a: kernel-reported I2C errors — near-zero false positives.
  t=$(now)
  if kernel_errors "$LAST_KMSG_CHECK" && (( t - LAST_KMSG >= KMSG_COOLDOWN )); then
    LAST_KMSG=$t
    recover "kernel i2c error"
  fi
  LAST_KMSG_CHECK=$t

  # Pace the loop to ~CYCLE beyond sampling time.
  elapsed=$(( $(now) - t_start ))
  (( elapsed < CYCLE )) && sleep $(( CYCLE - elapsed ))
done
EOF
}

# ── Detection (runs on the live system) ────────────────────────────────────────
find_devices() {
  for sysfs in /sys/bus/i2c/devices/*-*/; do
    [[ -e "$sysfs/modalias" ]] || continue
    grep -q "PNP0C50\|MSFT0001" "$sysfs/modalias" 2>/dev/null && echo "${sysfs%/}"
  done
  return 0
}

touchpad_detected() {
  [[ -n "$(find_devices)" ]]
}

# ── Install / upgrade (idempotent) ─────────────────────────────────────────────
install_fixes() {
  # Remove legacy over-engineered config from older installs (no-op on fresh systems)
  sudo rm -f /etc/udev/rules.d/99-touchpad-fix.rules \
              /etc/systemd/system/touchpad-fixer.service \
              /etc/systemd/system/touchpad-persist.service \
              /lib/systemd/system-sleep/touchpad-fix.sh \
              /usr/local/bin/touchpad-restart
  sudo systemctl disable --now touchpad-fixer.service touchpad-persist.service 2>/dev/null || true

  print_status "Installing touchpad-fixer..."
  fixer_script | sudo tee /usr/local/sbin/touchpad-fixer > /dev/null
  sudo chmod +x /usr/local/sbin/touchpad-fixer

  print_status "Installing touchpad-watchdog..."
  watchdog_script | sudo tee /usr/local/sbin/touchpad-watchdog > /dev/null
  sudo chmod +x /usr/local/sbin/touchpad-watchdog

  print_status "Installing resume service..."
  sudo tee /etc/systemd/system/touchpad-fixer-resume.service > /dev/null << 'EOF'
[Unit]
Description=Reset I2C-HID touchpad after suspend (power fix + unbind/rebind)
After=suspend.target hibernate.target hybrid-sleep.target
[Service]
Type=oneshot
ExecStart=/bin/bash -c 'sleep 2 && /usr/local/sbin/touchpad-fixer reset'
[Install]
WantedBy=suspend.target hibernate.target hybrid-sleep.target
EOF

  print_status "Installing watchdog service..."
  sudo tee /etc/systemd/system/touchpad-watchdog.service > /dev/null << 'EOF'
[Unit]
Description=Watchdog for I2C-HID touchpad auto-recovery
After=multi-user.target
[Service]
Type=simple
ExecStart=/usr/local/sbin/touchpad-watchdog
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
EOF

  sudo systemctl daemon-reload 2>/dev/null || true
  sudo systemctl enable --now touchpad-fixer-resume.service 2>/dev/null || true
  sudo systemctl enable --now touchpad-watchdog.service 2>/dev/null || true

  # Apply prevention immediately (don't wait for the watchdog's first cycle)
  sudo /usr/local/sbin/touchpad-fixer power 2>/dev/null || true

  echo ""
  print_ok "Installed: fixer + watchdog + resume service + runtime-PM prevention"
  print_ok "Removed:   legacy udev rule, boot/persist services, sleep hook"
}

# ── Quick restart ──────────────────────────────────────────────────────────────
quick_restart() {
  if [[ -x /usr/local/sbin/touchpad-fixer ]]; then
    /usr/local/sbin/touchpad-fixer reset
  else
    print_error "Fixer not installed — choose 'Apply' first."
  fi
}

# ── Status ─────────────────────────────────────────────────────────────────────
show_status() {
  echo "=== I2C-HID device state ==="
  if [[ -x /usr/local/sbin/touchpad-fixer ]]; then
    /usr/local/sbin/touchpad-fixer status
  else
    local found=0 sysfs drv
    for sysfs in $(find_devices); do
      found=1
      drv="NOT BOUND"
      [[ -L "$sysfs/driver" ]] && drv=$(basename "$(readlink -f "$sysfs/driver")")
      printf "  %-20s bind=%s\n" "$(basename "$sysfs")" "$drv"
    done
    [[ "$found" -eq 0 ]] && echo "  (no I2C-HID touchpad found)"
  fi
  echo ""
  echo "Services:"
  systemctl is-active touchpad-fixer-resume.service touchpad-watchdog.service 2>/dev/null
  echo ""
  echo "Recent watchdog activity (last 10 log lines):"
  journalctl -t touchpad-watchdog --no-pager -n 10 2>/dev/null || echo "  (no logs / journal not readable)"
}

# ── Uninstall ──────────────────────────────────────────────────────────────────
uninstall_fixes() {
  print_warning "This will remove all touchpad fixes, services, and scripts."
  read -p "Are you sure? (y/N): " -n 1 -r
  echo
  [[ $REPLY =~ ^[Yy]$ ]] || { print_status "Uninstall cancelled."; return; }

  sudo systemctl disable --now touchpad-fixer.service 2>/dev/null || true
  sudo systemctl disable --now touchpad-fixer-resume.service 2>/dev/null || true
  sudo systemctl disable --now touchpad-watchdog.service 2>/dev/null || true
  sudo systemctl disable --now touchpad-persist.service 2>/dev/null || true
  sudo rm -f /etc/systemd/system/touchpad-fixer.service \
              /etc/systemd/system/touchpad-fixer-resume.service \
              /etc/systemd/system/touchpad-watchdog.service \
              /etc/systemd/system/touchpad-persist.service
  sudo rm -f /lib/systemd/system-sleep/touchpad-fix.sh \
              /etc/udev/rules.d/99-touchpad-fix.rules \
              /usr/local/sbin/touchpad-fixer \
              /usr/local/sbin/touchpad-watchdog \
              /usr/local/bin/touchpad-restart
  sudo systemctl daemon-reload 2>/dev/null || true

  print_ok "Uninstall complete!"
}

# ── Menu ───────────────────────────────────────────────────────────────────────
show_menu() {
  print_header

  if touchpad_detected; then
    local sysfs drv
    sysfs=$(find_devices | head -1)
    drv="NOT BOUND"
    [[ -L "$sysfs/driver" ]] && drv=$(basename "$(readlink -f "$sysfs/driver")")
    echo -e "${GREEN}Touchpad detected:${NC} $(basename "$sysfs") (driver: $drv)"
  else
    echo -e "${RED}No I2C-HID touchpad detected${NC}"
  fi

  local installed=false
  [[ -f /usr/local/sbin/touchpad-fixer ]] && installed=true
  echo ""
  if [[ "$installed" == "true" ]]; then
    echo -e "${GREEN}Touchpad fixes are installed${NC}"
  else
    echo -e "${RED}Touchpad fixes are NOT installed${NC}"
  fi
  echo ""
  echo "What would you like to do?"
  echo "1) Apply/upgrade fixes"
  echo "2) Quick restart touchpad (when frozen)"
  echo "3) Check status"
  echo "4) Uninstall all fixes"
  echo "5) Exit"
  echo ""
}

require_root() {
  if [[ "$EUID" -ne 0 ]]; then
    print_warning "This action requires root privileges."
    print_status "Re-running with sudo..."
    exec sudo bash "$0" "$@"
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  case "${1:-}" in
    --apply)
      require_root "$@"
      install_fixes
      exit 0
      ;;
    --reset)
      require_root "$@"
      quick_restart
      exit 0
      ;;
    --uninstall)
      require_root "$@"
      uninstall_fixes
      exit 0
      ;;
    --status)
      show_status
      exit 0
      ;;
    --print-fixer)
      fixer_script
      exit 0
      ;;
    --print-watchdog)
      watchdog_script
      exit 0
      ;;
    "")
      # interactive menu below
      ;;
    *)
      echo "Usage: $0 [--apply|--reset|--uninstall|--status]"
      echo "Run without arguments for the interactive menu."
      exit 1
      ;;
  esac

  require_root "$@"

  while true; do
    show_menu
    read -p "Please enter your choice (1-5): " choice

    case $choice in
      1) install_fixes ;;
      2) quick_restart ;;
      3) show_status ;;
      4) uninstall_fixes ;;
      5) print_status "Goodbye!"; exit 0 ;;
      *) print_error "Invalid choice! Please enter 1-5." ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
    echo ""
  done
}

main "$@"
