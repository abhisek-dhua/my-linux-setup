#!/bin/bash

CRED_HELPER="/home/developer/.git-cred-helper"
CRED_CONFIG="/home/developer/.git-credentials.conf"

backup_config() {
  if [[ -f "$CRED_HELPER" ]]; then
    cp "$CRED_HELPER" "${CRED_HELPER}.bak.$(date +%Y%m%d%H%M%S)"
    echo "Backup: ${CRED_HELPER}.bak.*"
  fi
}

write_helper() {
  cat > "$CRED_HELPER" <<'HELPER'
#!/bin/bash
ACTION=$1
FILE=$(mktemp)
cat > "$FILE"

if [[ "$ACTION" == "get" ]]; then
  while read -r line; do
    [[ "$line" =~ ^([^=]+)=(.*)$ ]] && key="${BASH_REMATCH[1]}" value="${BASH_REMATCH[2]}"
    case "$key" in
      protocol) PROTO="$value" ;;
      host) HOST="$value" ;;
      path) PATH_VAL="$value" ;;
    esac
  done < "$FILE"

  CONF="/home/developer/.git-credentials.conf"
  if [[ -f "$CONF" ]]; then
    best_match=""
    best_score=0
    while IFS='|' read -r c_host c_user c_token c_path; do
      [[ "$c_host" != "$HOST" ]] && continue
      if [[ -n "$c_path" ]]; then
        if [[ "$PATH_VAL" == $c_path ]]; then
          score=${#c_path}
          if (( score > best_score )); then
            best_score=$score
            best_match="$c_user|$c_token"
          fi
        fi
      else
        if [[ -z "$best_match" ]]; then
          best_match="$c_user|$c_token"
        fi
      fi
    done < "$CONF"

    if [[ -n "$best_match" ]]; then
      IFS='|' read -r u p <<< "$best_match"
      echo "username=$u"
      echo "password=$p"
    fi
  fi
fi

rm "$FILE"
HELPER
  chmod +x "$CRED_HELPER"
}

show_accounts() {
  echo ""
  echo "=========================================="
  echo "   Git Multi-Account Credential Manager"
  echo "=========================================="
  echo ""

  if [[ -s "$CRED_CONFIG" ]]; then
    echo "Configured accounts:"
    echo "------------------------------------------"
    printf "%-4s %-15s %-20s %-30s %s\n" "#" "HOST" "USERNAME" "TOKEN" "PATH PATTERN"
    echo "------------------------------------------"
    i=1
    while IFS='|' read -r host user token path; do
      masked="${token:0:10}...${token: -6}"
      printf "%-4s %-15s %-20s %-30s %s\n" "$i" "$host" "$user" "$masked" "${path:-*}"
      ((i++))
    done < "$CRED_CONFIG"
    echo "------------------------------------------"
  else
    echo "No accounts configured yet."
  fi
  echo ""
  echo "1) Add GitHub account"
  echo "2) Add Bitbucket account"
  echo "3) Remove an account"
  echo "4) View full config"
  echo "5) Test a credential"
  echo "6) Exit"
  echo ""
  read -p "Choose option [1-6]: " choice
}

add_account() {
  local service="$1"
  echo ""
  echo "--- Add $service Account ---"
  read -p "Username: " username
  read -p "Token/Password: " token
  read -p "Path pattern (e.g., orgname/*) or leave empty for all: " path_pattern

  if [[ -z "$username" || -z "$token" ]]; then
    echo "Error: Username and token are required."
    return 1
  fi

  echo ""
  echo "Summary:"
  echo "  Host:  $service"
  echo "  User:  $username"
  echo "  Token: ${token:0:10}...${token: -6}"
  echo "  Path:  ${path_pattern:-* (all repos)}"
  echo ""
  read -p "Add this account? [y/N]: " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Cancelled."
    return 0
  fi

  echo "${service}|${username}|${token}|${path_pattern}" >> "$CRED_CONFIG"
  chmod 600 "$CRED_CONFIG"
  echo "Account added."
}

remove_account() {
  if [[ ! -s "$CRED_CONFIG" ]]; then
    echo "No accounts to remove."
    return
  fi

  local i=1
  while IFS='|' read -r host user token path; do
    printf "%-4s %-15s %-20s %s\n" "$i" "$host" "$user" "${path:-*}"
    ((i++))
  done < "$CRED_CONFIG"

  echo ""
  read -p "Enter number to remove: " num

  if [[ "$num" =~ ^[0-9]+$ ]] && (( num >= 1 && num < i )); then
    read -p "Remove account #$num? [y/N]: " confirm
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
      sed -i "${num}d" "$CRED_CONFIG"
      echo "Removed."
    fi
  else
    echo "Invalid selection."
  fi
}

test_credential() {
  echo ""
  read -p "Protocol [https]: " protocol
  protocol=${protocol:-https}
  read -p "Host [bitbucket.org]: " host
  host=${host:-bitbucket.org}
  read -p "Path: " path

  echo ""
  echo "Result:"
  echo -e "protocol=${protocol}\nhost=${host}\npath=${path}" | "$CRED_HELPER" get 2>&1
}

view_config() {
  echo ""
  echo "--- .git-credentials.conf ---"
  cat "$CRED_CONFIG" 2>/dev/null || echo "(empty)"
  echo ""
  echo "--- .git-cred-helper ---"
  cat "$CRED_HELPER"
}

# --- Main ---
backup_config
write_helper

while true; do
  show_accounts
  case $choice in
    1) add_account "github.com" ;;
    2) add_account "bitbucket.org" ;;
    3) remove_account ;;
    4) view_config ;;
    5) test_credential ;;
    6) echo "Done."; exit 0 ;;
    *) echo "Invalid option." ;;
  esac
done
