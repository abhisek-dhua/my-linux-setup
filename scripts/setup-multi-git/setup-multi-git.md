# Git Multi-Account Credential Helper

## Overview

This setup allows using **multiple Git accounts** (GitHub, Bitbucket) on the same machine with **path-based routing** — different tokens for different organizations/repositories, all automatic.

---

## File Structure

```
~/.gitconfig                    # Global git config (entry point)
~/.git-cred-helper             # Credential helper script
~/.git-credentials.conf        # Account database
~/setup-multi-git.sh           # Interactive setup script
```

---

## How the Flow Works

### Step 1: Git needs authentication

When you run `git clone`, `git push`, or `git pull` over HTTPS:

```
git clone https://bitbucket.org/bharatkarwaan/bhatkn_admin_api.git
```

Git sees it needs credentials for `bitbucket.org`.

### Step 2: Git reads `.gitconfig`

```ini
[credential]
    helper = /home/developer/.git-cred-helper
    useHttpPath = true
```

- `helper` — tells Git to call `.git-cred-helper` script for credentials
- `useHttpPath = true` — **critical**: tells Git to also send the repo path (e.g., `bharatkarwaan/bhatkn_admin_api.git`) along with host/protocol

### Step 3: Git calls the credential helper

Git invokes `.git-cred-helper get` and pipes in:

```
protocol=https
host=bitbucket.org
path=bharatkarwaan/bhatkn_admin_api.git
```

### Step 4: Helper reads `.git-credentials.conf`

The helper parses the piped input, then scans the config file:

```
# Format: host|username|token|path_pattern
github.com|abhisek-dhua|ghp_xxx|abhisek-dhua/*
github.com|abhisek-msspl|ghp_yyy|*
bitbucket.org|abhisek-msspl|ATATT_xxx|bharatkarwaan/*
bitbucket.org|0h5l1oascen|ATATT_yyy|*
```

### Step 5: Path matching logic

The helper finds the **best (longest) matching path pattern**:

| Request Path | Matched Entry | Username |
|---|---|---|
| `bharatkarwaan/bhatkn_admin_api.git` | `bharatkarwaan/*` | `abhisek-msspl` |
| `other-workspace/repo.git` | `*` (fallback) | `0h5l1oascen` |
| `abhisek-dhua/myrepo.git` (GitHub) | `abhisek-dhua/*` | `abhisek-dhua` |
| `random-org/repo.git` (GitHub) | `*` (fallback) | `abhisek-msspl` |

**Longest path pattern wins** — so `bharatkarwaan/*` beats `*` when the path starts with `bharatkarwaan/`.

### Step 6: Helper returns credentials

```bash
username=abhisek-msspl
password=ATATT_xxx...
```

### Step 7: Git authenticates

Git uses the returned username + token to authenticate with Bitbucket/GitHub. Done.

---

## Config File Format

`.git-credentials.conf` — pipe-delimited, one account per line:

```
host|username|token|path_pattern
```

| Field | Example | Description |
|---|---|---|
| `host` | `github.com` | Git host to match |
| `username` | `abhisek-msspl` | Account username |
| `token` | `ghp_xxx...` | Personal access token or app password |
| `path_pattern` | `bharatkarwaan/*` | Glob pattern for repo paths (`*` = all) |

---

## Why This Design?

| Problem | Solution |
|---|---|
| Multiple GitHub accounts | Path pattern matches `org/*` to pick the right token |
| Work vs personal repos | Different tokens for different Bitbucket workspaces |
| Token rotation | Edit `.git-credentials.conf` — no code changes needed |
| Adding new accounts | Run `setup-git-credentials.sh` — interactive, safe |
| Breaking existing config | Script backs up before any change |

---

## Commands

```bash
# Run the interactive setup
bash ~/setup-multi-git.sh

# Test credential helper manually
echo -e "protocol=https\nhost=bitbucket.org\npath=bharatkarwaan/repo.git" | ~/.git-cred-helper get

# View current accounts
cat ~/.git-credentials.conf

# View credential helper
cat ~/.git-cred-helper

# View git config
git config --global --list
```

---

## Security Notes

- Tokens are stored in plain text in `~/.git-credentials.conf`
- Set strict permissions: `chmod 600 ~/.git-credentials.conf`
- Never commit `.git-credentials.conf` to any repo
- The setup script creates timestamped backups before changes
