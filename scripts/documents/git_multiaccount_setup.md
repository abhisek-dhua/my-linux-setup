# Multi-Account Git Setup

This guide explains how to set up Git to use different accounts based on folder location.

## Accounts

| Folder                    | User            | Email            | Account              |
| ------------------------- | --------------- | ---------------- | -------------------- |
| `~/Documents/AI/`         | (Personal Name) | (Personal Email) | Personal (GitHub)    |
| `/var/www/html/Topnotch/` | (Topnotch Name) | (Topnotch Email) | Topnotch (BitBucket) |
| All other paths           | (Work Name)     | (Work Email)     | Work (GitHub)        |

## Files Created

```
~/.gitconfig                     - Global git configuration (Work)
~/.git-cred-helper              - Custom credential helper script
~/Documents/AI/.gitconfig      - Personal git configuration
/var/www/html/Topnotch/.gitconfig - Topnotch git configuration
```

## How It Works

1. Uses `credential.useHttpPath = true` to differentiate credentials by repository path
2. Custom credential helper (`~/.git-cred-helper`) returns the right token based on the repo path:
   - `Personal_GitHub_User/*` → Personal token
   - `bitbucket.org` → Topnotch token
   - Everything else → Work token

## Setup Files

### ~/.gitconfig

```ini
[user]
	name = WORK_NAME
	email = WORK_EMAIL

[credential]
	helper = /home/$USER/.git-cred-helper
	useHttpPath = true

[pager]
	branch = false

[includeIf "gitdir:~/Documents/AI/"]
	path = ~/Documents/AI/.gitconfig

# Only if Topnotch is configured:
[includeIf "gitdir:/var/www/html/Topnotch/"]
	path = /var/www/html/Topnotch/.gitconfig
```

### ~/Documents/AI/.gitconfig

```ini
[user]
	name = PERSONAL_NAME
	email = PERSONAL_EMAIL
```

### /var/www/html/Topnotch/.gitconfig

```ini
[user]
	name = TOPNOTCH_NAME
	email = TOPNOTCH_EMAIL
```

### ~/.git-cred-helper

```bash
#!/bin/bash

ACTION=$1
FILE=$(mktemp)

cat > $FILE

if [[ "$ACTION" == "get" ]]; then
  while read -r line; do
    [[ "$line" =~ ^([^=]+)=(.*)$ ]] && key="${BASH_REMATCH[1]}" value="${BASH_REMATCH[2]}"
    case "$key" in
      protocol) protocol="$value" ;;
      host) host="$value" ;;
      path) path="$value" ;;
    esac
  done < $FILE

  if [[ "$protocol" == "https" && "$host" == "github.com" ]]; then
    if [[ "$path" == PERSONAL_GITHUB_USER/* ]]; then
      echo "username=PERSONAL_GITHUB_USER"
      echo "password=TOKEN_PERSONAL"
    else
      echo "username=WORK_GITHUB_USER"
      echo "password=TOKEN_WORK"
    fi
  elif [[ "$protocol" == "https" && "$host" == "bitbucket.org" ]]; then
    echo "username=TOPNOTCH_KEY"
    echo "password=TOKEN_TOPNOTCH"
  fi
fi

rm $FILE
```

## Automated Setup

Run the setup script on any new system:

```bash
chmod +x ~/Documents/*/my-linux-setup/scripts/setup-git-multiaccount.sh
~/Documents/*/my-linux-setup/scripts/setup-git-multiaccount.sh
```

The script will prompt for all accounts:

**Work Account (always):**

- Work Git Name
- Work Git Email
- Work GitHub Username
- Work GitHub Token

**Personal Account (always):**

- Personal Git Name
- Personal Git Email
- Personal GitHub Username
- Personal GitHub Token

**Topnotch Account (optional):**

- Configure Topnotch? (y/n)
- If yes: Topnotch Git Name, Topnotch Git Email, Topnotch BitBucket Key, Topnotch BitBucket Token

## Testing

Test each account by pushing from the respective folder:

```bash
# Personal account test
cd ~/Documents/AI/Projects/project_folder
git push

# Work account test
cd ~/Abhisek/project_folder
git push

# Topnotch account test
cd /var/www/html/Topnotch/project_folder
git push
```

## Troubleshooting

If push fails with "Permission denied":

1. Check which credential is being used: `git credential fill`
2. Verify tokens in `~/.git-cred-helper` are correct
3. Ensure `useHttpPath = true` is set in `~/.gitconfig`
