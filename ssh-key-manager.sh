#!/bin/bash
set -euo pipefail

SSH_DIR="$HOME/.ssh"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SSH_BACKUP="$SCRIPT_DIR/ssh.zip"

mkdir -p "$SSH_DIR"
chmod 700 "$SSH_DIR"

# ============================================================
# 🔧 Helper Functions
# ============================================================
restore_keys() {
    if [[ ! -f "$SSH_BACKUP" ]]; then
        echo "❌ No ssh.zip found in script directory"
        return 1
    fi

    if ls "$SSH_DIR"/id_* &>/dev/null; then
        echo "⚠️ Existing keys found, backing up..."
        mkdir -p "$SSH_DIR/old_keys_$(date +%Y%m%d_%H%M%S)"
        mv "$SSH_DIR"/id_* "$SSH_DIR/old_keys_$(date +%Y%m%d_%H%M%S)"/ 2>/dev/null || true
    fi

    unzip -o "$SSH_BACKUP" -d "$SSH_DIR" >/dev/null 2>&1
    chmod 600 "$SSH_DIR"/id_* 2>/dev/null || true
    chmod 644 "$SSH_DIR"/*.pub 2>/dev/null || true
    echo "✅ SSH keys restored from backup"
}

create_new_key() {
    if ls "$SSH_DIR"/id_* &>/dev/null; then
        echo "⚠️ Existing keys found, backing up..."
        mkdir -p "$SSH_DIR/old_keys_$(date +%Y%m%d_%H%M%S)"
        mv "$SSH_DIR"/id_* "$SSH_DIR/old_keys_$(date +%Y%m%d_%H%M%S)"/ 2>/dev/null || true
    fi

    if [[ -z "${SSH_EMAIL:-}" ]]; then
        read -p "Enter email for SSH key: " SSH_EMAIL
    fi

    ssh-keygen -t ed25519 -C "${SSH_EMAIL}" -f "$SSH_DIR/id_ed25519" -N "" -q
    echo "✅ New SSH key generated"

    echo "📦 Creating backup to ssh.zip..."
    (cd "$SSH_DIR" && zip -q "$SSH_BACKUP" id_ed25519 id_ed25519.pub)
    echo "✅ Backup saved to $SSH_BACKUP"
}

backup_keys() {
    if ! ls "$SSH_DIR"/id_* &>/dev/null; then
        echo "❌ No SSH keys found to backup"
        return 1
    fi

    local priv_keys=()
    local pub_keys=()
    for f in "$SSH_DIR"/id_*; do
        if [[ "$f" == *.pub ]]; then
            pub_keys+=("$(basename "$f")")
        else
            priv_keys+=("$(basename "$f")")
        fi
    done

    (cd "$SSH_DIR" && zip -q "$SSH_BACKUP" "${priv_keys[@]}" "${pub_keys[@]}")
    echo "✅ Backup saved to $SSH_BACKUP"
}

view_public_key() {
    if ! ls "$SSH_DIR"/id_*.pub &>/dev/null; then
        echo "❌ No public key found"
        return 1
    fi

    echo ""
    echo "📋 Your public keys:"
    echo "===================="
    for f in "$SSH_DIR"/id_*.pub; do
        echo ""
        echo "📄 $(basename "$f"):"
        cat "$f"
    done
    echo ""
    echo "👉 Add to GitHub: https://github.com/settings/keys"
}

add_to_agent() {
    if command -v ssh-agent &>/dev/null; then
        eval "$(ssh-agent -s)" >/dev/null 2>&1
        ssh-add "$SSH_DIR"/id_* 2>/dev/null || true
        echo "✅ SSH key added to agent"
    fi
}

remove_keys() {
    if ! ls "$SSH_DIR"/id_* &>/dev/null; then
        echo "❌ No SSH keys found to remove"
        return 1
    fi

    echo "⚠️ The following keys will be removed:"
    ls -1 "$SSH_DIR"/id_*
    echo ""
    read -p "Are you sure? (y/n): " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "❌ Cancelled"
        return 1
    fi

    mkdir -p "$SSH_DIR/old_keys_$(date +%Y%m%d_%H%M%S)"
    mv "$SSH_DIR"/id_* "$SSH_DIR/old_keys_$(date +%Y%m%d_%H%M%S)"/ 2>/dev/null || true
    echo "✅ SSH keys removed and backed up"
}

# ============================================================
# 📋 Main Menu
# ============================================================
while true; do
    echo ""
    echo "🔑 SSH Key Manager"
    echo "==================="
    echo "1) Add SSH key (restore from backup or create new)"
    echo "2) Backup SSH keys"
    echo "3) View public key"
    echo "4) Remove SSH keys"
    echo "5) Exit"
    echo ""
    read -p "Choose an option [1-5]: " choice

    case $choice in
        1)
            echo ""
            echo "📦 Add SSH Key"
            echo "1) Restore from backup (ssh.zip)"
            echo "2) Create new SSH key"
            read -p "Choose [1-2]: " subchoice
            case $subchoice in
                1) restore_keys; add_to_agent ;;
                2) create_new_key; add_to_agent ;;
                *) echo "❌ Invalid option" ;;
            esac
            ;;
        2)
            echo ""
            backup_keys
            ;;
        3)
            echo ""
            view_public_key
            ;;
        4)
            echo ""
            remove_keys
            ;;
        5)
            echo "👋 Exiting SSH Key Manager"
            exit 0
            ;;
        *)
            echo "❌ Invalid option"
            ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
done
