#!/usr/bin/env bash
#
# cursor-auto-logout-toggle.sh
#
# Manage the system unit "cursor-auto-logout.service" and its shutdown hook.
#
#   disable  - permanently stop it from running (unit file kept, boot symlink removed)
#   enable   - permanently make it run again  (unit file kept, boot symlink restored)
#   remove   - delete the service completely  (unit file + shutdown hook, backup kept)
#   add      - (re)create the service from scratch (unit file + shutdown hook)
#   status   - show current state
#
# Everything needed to rebuild the service is embedded in this script, so
# "add" restores a byte-identical unit file and shutdown hook.
#
# If root privileges are required the script asks for the sudo password:
#   * on a normal terminal it uses the standard interactive sudo prompt
#   * without a terminal it reads the password itself and feeds it to sudo
#     (or you can export SUDO_PASSWORD=... for headless use)

set -uo pipefail

SERVICE="cursor-auto-logout.service"
UNIT_PATH="/etc/systemd/system/${SERVICE}"
HOOK_PATH="/usr/lib/systemd/system-shutdown/cursor-auto-logout"
HOOK_DIR="/usr/lib/systemd/system-shutdown"
WANTS_DIR="/etc/systemd/system/multi-user.target.wants"

SUDO_MODE=""     # "", "cached", "pw"
SUDO_PW=""
SUDO_ASKED=0
ACTION=""

cleanup() { SUDO_PW=""; }
trap cleanup EXIT

# ------------------------------------------------------------------ helpers --

info() { printf '\033[0;36m[INFO]\033[0m %s\n' "$*"; }
ok()   { printf '\033[0;32m[ OK ]\033[0m %s\n' "$*"; }
warn() { printf '\033[0;33m[WARN]\033[0m %s\n' "$*"; }
err()  { printf '\033[0;31m[ERR ]\033[0m %s\n' "$*" >&2; }
die()  { err "$*"; exit 1; }

unit_exists() { [ -f "$UNIT_PATH" ]; }
hook_exists() { [ -f "$HOOK_PATH" ]; }

target_user() {
  local u
  u="${SUDO_USER:-${USER:-}}"
  [ -n "$u" ] || u="$(id -un)"
  printf '%s\n' "$u"
}

target_home() {
  local u h
  u="$(target_user)"
  h="$(getent passwd "$u" 2>/dev/null | cut -d: -f6)"
  [ -n "$h" ] || h="$HOME"
  printf '%s\n' "$h"
}

backup_base() {
  local h="$1"
  if [ -d "${h}/Documents" ]; then
    printf '%s\n' "${h}/Documents"
  else
    printf '%s\n' "$h"
  fi
}

read_password() {
  local pw=""
  if [ -r /dev/tty ]; then
    printf 'sudo password: ' > /dev/tty 2>/dev/null || true
    IFS= read -r -s pw < /dev/tty 2>/dev/null || true
    printf '\n' > /dev/tty 2>/dev/null || true
  fi
  printf '%s' "$pw"
}

ensure_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    SUDO_MODE=""
    return 0
  fi
  if [ -n "$SUDO_MODE" ]; then
    return 0
  fi

  # already valid credentials?
  if sudo -n -v 2>/dev/null; then
    SUDO_MODE="cached"
    ok "Sudo credentials already valid."
    return 0
  fi

  if [ "$SUDO_ASKED" -eq 0 ]; then
    info "Administrator rights are required for this action."
    SUDO_ASKED=1
  fi

  # normal interactive terminal: let sudo ask the user itself
  if [ -t 0 ]; then
    info "Please enter your sudo password when prompted."
    if ! sudo -v; then
      die "Could not obtain sudo credentials. Aborting without any change."
    fi
    SUDO_MODE="cached"
    ok "Sudo credentials obtained."
    return 0
  fi

  # no terminal: ask here and hand the password to sudo
  local pw=""
  if [ -n "${SUDO_PASSWORD:-}" ]; then
    pw="$SUDO_PASSWORD"
  else
    pw="$(read_password)"
  fi
  if [ -z "$pw" ]; then
    die "No sudo password available (no terminal and SUDO_PASSWORD is unset). Aborting without any change."
  fi
  if ! printf '%s\n' "$pw" | sudo -S -p '' -v 2>/dev/null; then
    unset pw
    die "Wrong sudo password. Aborting without any change."
  fi
  SUDO_PW="$pw"
  unset pw
  SUDO_MODE="pw"
  ok "Sudo credentials obtained."
}

run_sudo() {
  case "$SUDO_MODE" in
    pw)     printf '%s\n' "$SUDO_PW" | sudo -S -p '' "$@" ;;
    cached) sudo "$@" ;;
    *)      "$@" ;;
  esac
}

verify_unit_intact() {
  if [ -f "$UNIT_PATH" ]; then
    ok "Unit file untouched and still present: ${UNIT_PATH}"
  else
    warn "Unit file was not found at ${UNIT_PATH}."
  fi
}

show_status() {
  local state active
  state="$(systemctl is-enabled "${SERVICE}" 2>&1)"
  active="$(systemctl is-active "${SERVICE}" 2>&1)"

  echo "-------------------------------------------------------------"
  printf 'Service        : %s\n' "${SERVICE}"
  printf 'Unit file      : %s\n' "$(unit_exists && echo "present (${UNIT_PATH})" || echo 'MISSING')"
  printf 'Shutdown hook  : %s\n' "$(hook_exists && echo "present (${HOOK_PATH})" || echo 'MISSING')"
  printf 'is-enabled     : %s\n' "${state}"
  printf 'is-active      : %s\n' "${active}"
  printf 'Owner of unit  : %s\n' "$(stat -c '%U:%G %a' "$UNIT_PATH" 2>/dev/null || echo '-')"
  printf 'Owner of hook  : %s\n' "$(stat -c '%U:%G %a' "$HOOK_PATH" 2>/dev/null || echo '-')"
  echo "Boot symlinks  :"
  if ls -l "${WANTS_DIR}" 2>/dev/null | grep -i cursor; then
    :
  else
    echo "  (none - service will not be started at boot)"
  fi
  echo "-------------------------------------------------------------"
}

# ------------------------------------------------------- embedded artefacts --

write_unit_template() {
  # $1 = destination file (writable)
  cat > "$1" <<'UNIT_EOF'
[Unit]
Description=Cursor logout before system shutdown

DefaultDependencies=no

Before=shutdown.target reboot.target halt.target poweroff.target umount.target
After=local-fs.target

Conflicts=shutdown.target

[Service]
Type=oneshot
User=root
ExecStart=/usr/lib/systemd/system-shutdown/cursor-auto-logout
TimeoutStartSec=15
KillMode=process

[Install]
WantedBy=multi-user.target
UNIT_EOF
}

write_hook_template() {
  # $1 = destination file (writable)
  cat > "$1" <<'HOOK_EOF'
#!/bin/bash

LOG="/var/log/cursor-auto-logout.log"

exec >>"$LOG" 2>&1

echo
echo "========================================"
echo "Cursor shutdown hook START"
echo "Date: $(date)"
echo "PID: $$"
echo "========================================"

USER_NAME="__CURSOR_USER__"
USER_HOME="__CURSOR_HOME__"

export HOME="$USER_HOME"

SQLITE="/usr/bin/sqlite3"

echo "[1] Checking sqlite3..."

if [ ! -x "$SQLITE" ]; then
    echo "ERROR: sqlite3 not found"
    exit 1
fi

echo "[2] Stopping Cursor..."

pgrep -u "$USER_NAME" -x cursor 2>/dev/null |
    xargs -r kill -TERM

pgrep -u "$USER_NAME" -x Cursor 2>/dev/null |
    xargs -r kill -TERM

sleep 2

echo "[3] Clearing Cursor desktop authentication..."

CURSOR_DB="$HOME/.config/Cursor/User/globalStorage/state.vscdb"

if [ -f "$CURSOR_DB" ]; then

    "$SQLITE" "$CURSOR_DB" <<'SQL'

DELETE FROM ItemTable
WHERE key IN (
    'cursorAuth/accessToken',
    'cursorAuth/refreshToken',
    'cursorAuth/cachedEmail',
    'cursorAuth/cachedScopedProfile',
    'cursorAuth/cachedSignUpType',
    'cursorAuth/stripeMembershipAuthId',
    'cursorAuth/stripeMembershipType',
    'cursorAuth/stripeSubscriptionStatus',
    'adminSettings.cachedAuthId',
    'glass.lastSignedInAuthId'
);

SQL

    echo "Cursor DB cleaned."

else
    echo "Cursor DB not found:"
    echo "$CURSOR_DB"
fi

echo "[4] Clearing Cursor embedded browser..."

CURSOR_BROWSER="$HOME/.config/Cursor/Partitions/cursor-browser"

if [ -d "$CURSOR_BROWSER" ]; then

    rm -f \
        "$CURSOR_BROWSER/Cookies" \
        "$CURSOR_BROWSER/Cookies-journal" \
        "$CURSOR_BROWSER/Cookies-shm" \
        "$CURSOR_BROWSER/Cookies-wal"

    rm -rf \
        "$CURSOR_BROWSER/Local Storage" \
        "$CURSOR_BROWSER/Session Storage" \
        "$CURSOR_BROWSER/WebStorage"

    echo "Cursor embedded browser cleaned."

else
    echo "Cursor embedded browser directory not found."
fi

echo "[5] Clearing Cursor cookies from Chrome profiles..."

CHROME_ROOTS=(
    "$HOME/.config/google-chrome"
    "$HOME/.config/google-chrome-beta"
    "$HOME/.config/google-chrome-unstable"
    "$HOME/.config/chromium"
)

for ROOT in "${CHROME_ROOTS[@]}"; do

    [ -d "$ROOT" ] || continue

    echo "Checking Chrome root: $ROOT"

    find "$ROOT" \
        -maxdepth 2 \
        -type f \
        -name "Cookies" \
        -print0 2>/dev/null |
    while IFS= read -r -d '' COOKIE_DB; do

        echo "Checking cookie DB: $COOKIE_DB"

        if ! "$SQLITE" "$COOKIE_DB" \
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cookies' LIMIT 1;" \
            2>/dev/null | grep -q '^1$'; then

            echo "Not a valid Chrome cookie DB. Skipping."
            continue
        fi

        "$SQLITE" "$COOKIE_DB" <<'SQL'

DELETE FROM cookies
WHERE host_key LIKE '%cursor.com%'
   OR host_key LIKE '%cursor.sh%'
   OR host_key LIKE '%authenticator.cursor.sh%';

SQL

        echo "Cursor cookies removed from: $COOKIE_DB"

    done

done

echo "========================================"
echo "Cursor shutdown hook FINISHED"
echo "Date: $(date)"
echo "========================================"

exit 0
HOOK_EOF
}

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'
}

build_hook_file() {
  # $1 = destination file, $2 = user, $3 = home
  local dest="$1" u="$2" h="$3" tmp eu eh
  tmp="$(mktemp)" || die "mktemp failed"
  write_hook_template "$tmp"
  eu="$(escape_sed_replacement "$u")"
  eh="$(escape_sed_replacement "$h")"
  sed -i -e "s|__CURSOR_USER__|${eu}|g" -e "s|__CURSOR_HOME__|${eh}|g" "$tmp"
  cat "$tmp" > "$dest"
  rm -f "$tmp"
}

# ----------------------------------------------------------------- actions ---

do_disable() {
  ensure_sudo
  info "Disabling ${SERVICE} permanently (removes the boot symlink, stops it if running)..."

  if ! run_sudo systemctl disable --now "${SERVICE}"; then
    die "systemctl disable failed - nothing was changed."
  fi
  run_sudo systemctl reset-failed "${SERVICE}" >/dev/null 2>&1 || true

  local state
  state="$(systemctl is-enabled "${SERVICE}" 2>&1)"
  if [ "$state" = "disabled" ] || [ "$state" = "static" ] || [ "$state" = "masked" ]; then
    ok "${SERVICE} is now ${state} and will not run at boot or on shutdown."
  else
    warn "Expected 'disabled' but systemctl reports '${state}'. Please check manually."
  fi

  verify_unit_intact
  show_status
}

do_enable() {
  ensure_sudo
  info "Enabling ${SERVICE} permanently (re-creates the boot symlink)..."

  if ! run_sudo systemctl enable "${SERVICE}"; then
    die "systemctl enable failed - nothing was changed."
  fi

  local state
  state="$(systemctl is-enabled "${SERVICE}" 2>&1)"
  if [ "$state" = "enabled" ]; then
    ok "${SERVICE} is enabled and will run at boot / before shutdown."
  else
    warn "Expected 'enabled' but systemctl reports '${state}'. Please check manually."
  fi

  verify_unit_intact

  local answer=""
  printf 'Also START the unit right now? (y/N) '
  read -r answer
  case "$answer" in
    [yY]|[yY][eE][sS])
      warn "Starting runs ${HOOK_PATH} immediately."
      if run_sudo systemctl start "${SERVICE}"; then
        ok "Unit started."
      else
        warn "Unit could not be started (it will still run at boot)."
      fi
      ;;
    *)
      info "Skipped immediate start; the unit stays enabled for the next boot."
      ;;
  esac

  show_status
}

do_remove() {
  ensure_sudo

  local answer=""
  printf 'This DELETES %s and %s permanently.\n' "$UNIT_PATH" "$HOOK_PATH"
  printf 'Type REMOVE to confirm: '
  read -r answer
  if [ "$answer" != "REMOVE" ]; then
    info "Aborted - nothing was changed."
    return 0
  fi

  local bdir bbase
  bbase="$(backup_base "$(target_home)")"
  bdir="${bbase}/cursor-auto-logout-backup-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$bdir" 2>/dev/null || bdir="$(mktemp -d)"

  if [ -f "$UNIT_PATH" ]; then
    run_sudo cp "$UNIT_PATH" "${bdir}/$(basename "$UNIT_PATH")" 2>/dev/null || true
  fi
  if [ -f "$HOOK_PATH" ]; then
    run_sudo cp "$HOOK_PATH" "${bdir}/$(basename "$HOOK_PATH")" 2>/dev/null || true
  fi
  run_sudo chown -R "$(target_user):$(id -gn)" "$bdir" 2>/dev/null || true
  run_sudo chmod -R u+rw "$bdir" 2>/dev/null || true
  ok "Backup saved to: ${bdir}"

  info "Removing the service..."
  run_sudo systemctl disable --now "${SERVICE}" >/dev/null 2>&1 || true
  run_sudo systemctl stop "${SERVICE}" >/dev/null 2>&1 || true
  run_sudo rm -f "${WANTS_DIR}/${SERVICE}" 2>/dev/null || true
  run_sudo rm -f "$UNIT_PATH" 2>/dev/null || true
  run_sudo rm -f "$HOOK_PATH" 2>/dev/null || true
  run_sudo systemctl reset-failed "${SERVICE}" >/dev/null 2>&1 || true
  run_sudo systemctl daemon-reload || true

  if unit_exists || hook_exists; then
    die "Removal incomplete - one or more files are still present. Check: ${bdir}"
  fi

  ok "${SERVICE} and its shutdown hook have been removed."
  ok "To bring it back, run this script with: add"
  show_status
}

do_add() {
  ensure_sudo

  local answer=""
  if [ -f "$UNIT_PATH" ] || [ -f "$HOOK_PATH" ]; then
    printf '%s and/or %s already exist.\n' "$UNIT_PATH" "$HOOK_PATH"
    printf 'Overwrite them with the embedded copies? (y/N) '
    read -r answer
    case "$answer" in
      [yY]|[yY][eE][sS]) : ;;
      *) info "Aborted - nothing was changed."; return 0 ;;
    esac
  fi

  local u h tmp_unit tmp_hook
  u="$(target_user)"
  h="$(target_home)"
  info "Rebuilding the service for user '${u}' (home ${h})..."

  tmp_unit="$(mktemp)" || die "mktemp failed"
  tmp_hook="$(mktemp)" || die "mktemp failed"
  write_unit_template "$tmp_unit"
  build_hook_file "$tmp_hook" "$u" "$h"

  run_sudo install -D -m 644 -o root -g root "$tmp_unit" "$UNIT_PATH" || die "could not install ${UNIT_PATH}"
  run_sudo install -D -m 700 -o root -g root "$tmp_hook" "$HOOK_PATH" || die "could not install ${HOOK_PATH}"
  rm -f "$tmp_unit" "$tmp_hook"

  run_sudo systemctl daemon-reload || true
  ok "Unit file written    : ${UNIT_PATH}"
  ok "Shutdown hook written: ${HOOK_PATH}"

  info "Enabling the service..."
  if ! run_sudo systemctl enable "${SERVICE}"; then
    warn "systemctl enable failed - the files are in place, enable it manually."
  else
    ok "${SERVICE} is enabled."
  fi

  local state
  state="$(systemctl is-enabled "${SERVICE}" 2>&1)"
  [ "$state" = "enabled" ] || warn "is-enabled reports '${state}'."

  show_status
}

interactive_menu() {
  while true; do
    echo
    echo "=== cursor-auto-logout.service manager ==="
    echo "  1) Disable permanently  (keep files, remove boot symlink)"
    echo "  2) Enable permanently   (keep files, restore boot symlink)"
    echo "  3) Remove the service   (delete unit + hook, backup kept)"
    echo "  4) Add the service      (recreate unit + hook from this script)"
    echo "  5) Status"
    echo "  6) Quit"
    echo "--------------------------------------------"
    printf 'Choose [1-6]: '
    local choice
    read -r choice
    case "$choice" in
      1) do_disable; return 0 ;;
      2) do_enable;  return 0 ;;
      3) do_remove;  return 0 ;;
      4) do_add;     return 0 ;;
      5) show_status; return 0 ;;
      6|q|Q) info "Bye."; return 0 ;;
      *) warn "Invalid choice." ;;
    esac
  done
}

usage() {
  cat <<'USAGE_EOF'
cursor-auto-logout-toggle.sh - manage cursor-auto-logout.service

Usage:
  ./cursor-auto-logout-toggle.sh              interactive menu
  ./cursor-auto-logout-toggle.sh disable      permanently disable (files kept)
  ./cursor-auto-logout-toggle.sh enable       permanently enable  (files kept)
  ./cursor-auto-logout-toggle.sh remove       delete the service (backup kept)
  ./cursor-auto-logout-toggle.sh add          recreate the service from scratch
  ./cursor-auto-logout-toggle.sh status       show current state

The sudo password is requested automatically when needed.
For headless use you may export SUDO_PASSWORD=yourpassword beforehand.
USAGE_EOF
}

main() {
  ACTION="${1:-}"
  case "$ACTION" in
    disable|disable-permanently|off) do_disable ;;
    enable|enable-permanently|on)   do_enable ;;
    remove|delete|rm)                do_remove ;;
    add|create|install)              do_add ;;
    status|show)                    show_status ;;
    "")                              interactive_menu ;;
    -h|--help|help)                  usage ;;
    *)
      err "Unknown option: ${ACTION}"
      usage
      exit 2
      ;;
  esac
}

main "$@"
