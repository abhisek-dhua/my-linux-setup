#!/bin/bash

echo "=== Multi-Account Git Setup Script ==="
echo ""

PERSONAL_NAME=""
PERSONAL_EMAIL=""
PERSONAL_GITHUB_USER=""
PERSONAL_TOKEN=""

WORK_NAME=""
WORK_EMAIL=""
WORK_GITHUB_USER=""
WORK_TOKEN=""

echo "=== Personal Account ==="
echo "Enter Personal Git Name: "
read PERSONAL_NAME
echo "Enter Personal Git Email: "
read PERSONAL_EMAIL
echo "Enter Personal GitHub Username: "
read PERSONAL_GITHUB_USER
echo "Enter Personal GitHub Token: "
read PERSONAL_TOKEN
echo ""

echo "=== Work Account ==="
echo "Enter Work Git Name: "
read WORK_NAME
echo "Enter Work Git Email: "
read WORK_EMAIL
echo "Enter Work GitHub Username: "
read WORK_GITHUB_USER
echo "Enter Work GitHub Token: "
read WORK_TOKEN
echo ""

echo "Setting up..."

USER=$(whoami)

cat > ~/.gitconfig << EOF
[user]
	name = WORK_NAME
	email = WORK_EMAIL

[pager]
	branch = false

[credential]
	helper = /home/$USER/.git-cred-helper
	useHttpPath = true

[includeIf "gitdir:~/Documents/Projects/Personal/"]
	path = ~/Documents/Projects/Personal/.gitconfig
EOF

mkdir -p ~/Documents/Projects/Work
mkdir -p ~/Documents/Projects/Personal
cat > ~/Documents/Projects/Personal/.gitconfig << EOF
[user]
	name = PERSONAL_NAME
	email = PERSONAL_EMAIL
EOF

cat > ~/.git-cred-helper << CRED_EOF
#!/bin/bash

ACTION=\$1
FILE=\$(mktemp)

cat > \$FILE

if [[ "\$ACTION" == "get" ]]; then
  while read -r line; do
    [[ "\$line" =~ ^([^=]+)=(.*)$ ]] && key="\${BASH_REMATCH[1]}" value="\${BASH_REMATCH[2]}"
    case "\$key" in
      protocol) protocol="\$value" ;;
      host) host="\$value" ;;
      path) path="\$value" ;;
    esac
  done < \$FILE

  if [[ "\$protocol" == "https" && "\$host" == "github.com" ]]; then
    if [[ "\$path" == PERSONAL_GITHUB_USER/* ]]; then
      echo "username=PERSONAL_GITHUB_USER"
      echo "password=TOKEN_PERSONAL"
    else
      echo "username=WORK_GITHUB_USER"
      echo "password=TOKEN_WORK"
    fi
  fi
fi

rm \$FILE
CRED_EOF

sed -i "s/WORK_NAME/$WORK_NAME/" ~/.gitconfig
sed -i "s/WORK_EMAIL/$WORK_EMAIL/" ~/.gitconfig
sed -i "s/WORK_GITHUB_USER/$WORK_GITHUB_USER/" ~/.git-cred-helper
sed -i "s/TOKEN_WORK/$WORK_TOKEN/" ~/.git-cred-helper

sed -i "s/PERSONAL_NAME/$PERSONAL_NAME/" ~/Documents/Projects/Personal/.gitconfig
sed -i "s/PERSONAL_EMAIL/$PERSONAL_EMAIL/" ~/Documents/Projects/Personal/.gitconfig
sed -i "s/PERSONAL_GITHUB_USER/$PERSONAL_GITHUB_USER/" ~/.git-cred-helper
sed -i "s/TOKEN_PERSONAL/$PERSONAL_TOKEN/" ~/.git-cred-helper

chmod +x ~/.git-cred-helper

echo "✓ Setup complete!"
echo ""
echo "Folder mapping:"
echo "  ~/Documents/Projects/Personal/*      → $PERSONAL_GITHUB_USER"
echo "  Other paths           → $WORK_GITHUB_USER"