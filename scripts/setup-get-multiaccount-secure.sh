#!/bin/bash

echo "=== Secure Multi-Account Git Setup ==="

# ===== INPUT =====
read -p "Personal Name: " PERSONAL_NAME
read -p "Personal Email: " PERSONAL_EMAIL
read -p "Work Name: " WORK_NAME
read -p "Work Email: " WORK_EMAIL

# ===== INSTALL SECURE CREDENTIAL HELPER =====
echo "Setting credential helper..."

git config --global credential.helper cache
# OR (better if available)
# git config --global credential.helper libsecret

# ===== SAFE GLOBAL CONFIG (append, not overwrite) =====
echo "Configuring global git..."

git config --global user.name "$WORK_NAME"
git config --global user.email "$WORK_EMAIL"

git config --global includeIf.gitdir:~/Projects/Personal/.path ~/Projects/Personal/.gitconfig

# ===== CREATE FOLDERS =====
mkdir -p ~/Projects/Personal
mkdir -p ~/Projects/Work

# ===== PERSONAL CONFIG =====
cat > ~/Projects/Personal/.gitconfig <<EOF
[user]
    name = $PERSONAL_NAME
    email = $PERSONAL_EMAIL
EOF

echo "✓ Config created"

# ===== INSTRUCTIONS =====
echo ""
echo "👉 Next Steps:"
echo "1. Clone using HTTPS:"
echo "   https://github.com/username/repo.git"
echo ""
echo "2. On first push, enter:"
echo "   - Username"
echo "   - Personal Access Token (NOT password)"
echo ""
echo "3. Git will securely cache it"

echo ""
echo "✅ Setup complete (secure version)"
