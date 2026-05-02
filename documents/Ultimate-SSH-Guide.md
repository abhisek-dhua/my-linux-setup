# Ultimate SSH Guide for Linux

## Table of Contents

1. [Introduction to SSH](#introduction-to-ssh)
2. [SSH Installation and Setup](#ssh-installation-and-setup)
3. [SSH Configuration](#ssh-configuration)
4. [SSH Key Management](#ssh-key-management)
5. [SSH Authentication Methods](#ssh-authentication-methods)
6. [SSH Connection and Usage](#ssh-connection-and-usage)
7. [SSH Tunneling and Port Forwarding](#ssh-tunneling-and-port-forwarding)
8. [SSH File Transfer](#ssh-file-transfer)
9. [SSH Security Best Practices](#ssh-security-best-practices)
10. [SSH Troubleshooting](#ssh-troubleshooting)
11. [Advanced SSH Features](#advanced-ssh-features)

---

## Introduction to SSH

### What is SSH?

SSH (Secure Shell) is a cryptographic network protocol for secure communication between computers. It provides:

- **Secure remote login**: Access remote systems securely
- **File transfer**: Copy files between systems
- **Port forwarding**: Tunnel other protocols through SSH
- **Key-based authentication**: More secure than passwords

### SSH Components

- **SSH Client**: `ssh` command for connecting to remote hosts
- **SSH Server**: `sshd` daemon that accepts connections
- **SSH Keys**: Public/private key pairs for authentication
- **SSH Agent**: Manages SSH keys in memory

---

## SSH Installation and Setup

### Installing SSH Client and Server

#### Ubuntu/Debian

```bash
# Install SSH client and server
sudo apt update
sudo apt install openssh-client openssh-server

# Start SSH service
sudo systemctl start ssh
sudo systemctl enable ssh

# Check status
sudo systemctl status ssh
```

#### CentOS/RHEL/Fedora

```bash
# Install SSH client and server
sudo yum install openssh-clients openssh-server
# or for newer versions:
sudo dnf install openssh-clients openssh-server

# Start SSH service
sudo systemctl start sshd
sudo systemctl enable sshd

# Check status
sudo systemctl status sshd
```

#### Arch Linux

```bash
# Install SSH
sudo pacman -S openssh

# Start SSH service
sudo systemctl start sshd
sudo systemctl enable sshd
```

### Basic SSH Server Configuration

#### Main Configuration File

```bash
# Edit SSH server configuration
sudo nano /etc/ssh/sshd_config
```

#### Essential Configuration Options

```bash
# Port (default: 22)
Port 22

# Listen on specific interface
ListenAddress 192.168.1.100

# Protocol version (use 2 for security)
Protocol 2

# Root login (disable for security)
PermitRootLogin no

# Password authentication
PasswordAuthentication yes

# Public key authentication
PubkeyAuthentication yes

# Maximum authentication attempts
MaxAuthTries 3

# Login grace time
LoginGraceTime 60

# Allow specific users
AllowUsers username1 username2

# Deny specific users
DenyUsers baduser

# Log level
LogLevel INFO
```

---

## SSH Configuration

### Client Configuration

#### Global Configuration

```bash
# Edit global SSH client configuration
sudo nano /etc/ssh/ssh_config
```

#### User Configuration

```bash
# Create user SSH configuration
mkdir -p ~/.ssh
nano ~/.ssh/config
```

#### SSH Config File Examples

```bash
# Basic host configuration
Host myserver
    HostName 192.168.1.100
    User myusername
    Port 22

# Multiple hosts with shared settings
Host *.example.com
    User admin
    IdentityFile ~/.ssh/id_rsa_admin

# Specific host with custom settings
Host production
    HostName prod.example.com
    User deploy
    Port 2222
    IdentityFile ~/.ssh/id_rsa_prod
    ForwardAgent yes
    Compression yes

# Jump host configuration
Host internal
    HostName 10.0.0.100
    User internal_user
    ProxyJump bastion.example.com
```

### Server Configuration Best Practices

#### Security Hardening

```bash
# Disable root login
PermitRootLogin no

# Disable password authentication (use keys only)
PasswordAuthentication no

# Disable empty passwords
PermitEmptyPasswords no

# Set maximum authentication attempts
MaxAuthTries 3

# Set login grace time
LoginGraceTime 30

# Disable X11 forwarding (if not needed)
X11Forwarding no

# Disable agent forwarding (if not needed)
AllowAgentForwarding no

# Set allowed users
AllowUsers username1 username2

# Set allowed groups
AllowGroups sshusers

# Use specific SSH key types
PubkeyAcceptedKeyTypes +ssh-rsa
```

#### Performance Tuning

```bash
# Enable compression
Compression yes

# Set connection keepalive
ClientAliveInterval 60
ClientAliveCountMax 3

# Enable TCP keepalive
TCPKeepAlive yes

# Set maximum sessions
MaxSessions 10
```

---

## SSH Key Management

### Generating SSH Keys

#### RSA Keys

```bash
# Generate RSA key pair
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Generate with specific filename
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_custom

# Generate without passphrase (less secure)
ssh-keygen -t rsa -b 4096 -N ""
```

#### Ed25519 Keys (Recommended)

```bash
# Generate Ed25519 key pair (more secure, smaller)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Generate with specific filename
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_custom
```

#### ECDSA Keys

```bash
# Generate ECDSA key pair
ssh-keygen -t ecdsa -b 521 -C "your_email@example.com"
```

### Managing SSH Keys

#### Copying Public Keys

```bash
# Copy public key to clipboard (Linux)
cat ~/.ssh/id_rsa.pub | xclip -selection clipboard

# Copy public key to clipboard (macOS)
cat ~/.ssh/id_rsa.pub | pbcopy

# Copy public key to remote server
ssh-copy-id username@remote_host

# Copy specific key to remote server
ssh-copy-id -i ~/.ssh/id_rsa_custom.pub username@remote_host

# Manual copy to remote server
cat ~/.ssh/id_rsa.pub | ssh username@remote_host "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```

#### Managing Multiple Keys

```bash
# List all SSH keys
ls -la ~/.ssh/

# View public key
cat ~/.ssh/id_rsa.pub

# View private key (be careful!)
cat ~/.ssh/id_rsa

# Change key permissions (important for security)
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub
chmod 700 ~/.ssh/
```

### SSH Agent

#### Starting SSH Agent

```bash
# Start SSH agent
eval "$(ssh-agent -s)"

# Add key to agent
ssh-add ~/.ssh/id_rsa

# Add all keys to agent
ssh-add

# List loaded keys
ssh-add -l

# Remove key from agent
ssh-add -d ~/.ssh/id_rsa

# Remove all keys from agent
ssh-add -D
```

#### Persistent SSH Agent

```bash
# Add to ~/.bashrc or ~/.zshrc
if [ -z "$SSH_AUTH_SOCK" ]; then
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/id_rsa
fi
```

---

## SSH Authentication Methods

### Password Authentication

```bash
# Connect with password (interactive)
ssh username@remote_host

# Connect with password (non-interactive, less secure)
sshpass -p 'password' ssh username@remote_host
```

### Key-Based Authentication

```bash
# Connect with default key
ssh username@remote_host

# Connect with specific key
ssh -i ~/.ssh/id_rsa_custom username@remote_host

# Connect with multiple keys
ssh -i ~/.ssh/id_rsa_custom -i ~/.ssh/id_rsa_backup username@remote_host
```

### Two-Factor Authentication

#### Google Authenticator Setup

```bash
# Install Google Authenticator
sudo apt install libpam-google-authenticator

# Configure for a user
google-authenticator

# Edit SSH PAM configuration
sudo nano /etc/pam.d/sshd

# Add this line:
auth required pam_google_authenticator.so
```

#### SSH Configuration for 2FA

```bash
# Edit SSH config
sudo nano /etc/ssh/sshd_config

# Enable challenge-response authentication
ChallengeResponseAuthentication yes
UsePAM yes

# Restart SSH service
sudo systemctl restart ssh
```

---

## SSH Connection and Usage

### Basic SSH Connection

```bash
# Connect to remote host
ssh username@remote_host

# Connect with specific port
ssh -p 2222 username@remote_host

# Connect with verbose output
ssh -v username@remote_host

# Connect with extra verbose output
ssh -vv username@remote_host

# Connect with maximum verbosity
ssh -vvv username@remote_host
```

### SSH Connection Options

```bash
# Connect and execute command
ssh username@remote_host "ls -la"

# Connect and execute multiple commands
ssh username@remote_host "cd /tmp && ls -la && pwd"

# Connect with specific cipher
ssh -c aes256-ctr username@remote_host

# Connect with specific MAC algorithm
ssh -m hmac-sha2-256 username@remote_host

# Connect with specific key exchange algorithm
ssh -o KexAlgorithms=diffie-hellman-group14-sha256 username@remote_host
```

### SSH Session Management

```bash
# Background SSH session
ssh -f -N username@remote_host

# Keep connection alive
ssh -o ServerAliveInterval=60 username@remote_host

# Set connection timeout
ssh -o ConnectTimeout=10 username@remote_host

# Exit on connection failure
ssh -o ExitOnForwardFailure=yes username@remote_host
```

---

## SSH Tunneling and Port Forwarding

### Local Port Forwarding

```bash
# Forward local port to remote host
ssh -L 8080:localhost:80 username@remote_host

# Forward local port to remote service
ssh -L 3306:database_host:3306 username@remote_host

# Forward with specific local interface
ssh -L 127.0.0.1:8080:localhost:80 username@remote_host

# Forward with specific remote interface
ssh -L 8080:192.168.1.100:80 username@remote_host
```

### Remote Port Forwarding

```bash
# Forward remote port to local host
ssh -R 8080:localhost:80 username@remote_host

# Forward remote port to local service
ssh -R 3306:localhost:3306 username@remote_host

# Forward with specific remote interface
ssh -R 0.0.0.0:8080:localhost:80 username@remote_host
```

### Dynamic Port Forwarding (SOCKS Proxy)

```bash
# Create SOCKS proxy on local port
ssh -D 1080 username@remote_host

# Create SOCKS proxy with specific interface
ssh -D 127.0.0.1:1080 username@remote_host
```

### Advanced Tunneling

```bash
# Multiple port forwards
ssh -L 8080:localhost:80 -L 3306:localhost:3306 username@remote_host

# Tunnel with compression
ssh -C -L 8080:localhost:80 username@remote_host

# Tunnel with background execution
ssh -f -N -L 8080:localhost:80 username@remote_host

# Tunnel through jump host
ssh -J username@jump_host -L 8080:localhost:80 username@target_host
```

---

## SSH File Transfer

### SCP (Secure Copy)

```bash
# Copy file from local to remote
scp local_file.txt username@remote_host:/remote/path/

# Copy file from remote to local
scp username@remote_host:/remote/file.txt ./

# Copy directory recursively
scp -r local_directory/ username@remote_host:/remote/path/

# Copy with specific port
scp -P 2222 local_file.txt username@remote_host:/remote/path/

# Copy with compression
scp -C local_file.txt username@remote_host:/remote/path/

# Copy with verbose output
scp -v local_file.txt username@remote_host:/remote/path/

# Copy multiple files
scp file1.txt file2.txt username@remote_host:/remote/path/
```

### RSYNC over SSH

```bash
# Sync files over SSH
rsync -av local_directory/ username@remote_host:/remote/path/

# Sync with compression
rsync -avz local_directory/ username@remote_host:/remote/path/

# Sync with progress
rsync -av --progress local_directory/ username@remote_host:/remote/path/

# Sync with exclude patterns
rsync -av --exclude='*.tmp' --exclude='*.log' local_directory/ username@remote_host:/remote/path/

# Sync with delete (mirror)
rsync -av --delete local_directory/ username@remote_host:/remote/path/

# Sync with bandwidth limit
rsync -av --bwlimit=1000 local_directory/ username@remote_host:/remote/path/
```

### SFTP (SSH File Transfer Protocol)

```bash
# Connect to SFTP server
sftp username@remote_host

# SFTP commands:
# get remote_file local_file
# put local_file remote_file
# mget remote_files*
# mput local_files*
# ls
# lls (local ls)
# cd remote_directory
# lcd local_directory
# mkdir remote_directory
# rmdir remote_directory
# rm remote_file
# quit
```

### SSHFS (SSH File System)

```bash
# Install SSHFS
sudo apt install sshfs  # Ubuntu/Debian
sudo yum install fuse-sshfs  # CentOS/RHEL

# Mount remote directory
sshfs username@remote_host:/remote/path /local/mount/point

# Mount with specific options
sshfs -o allow_other,default_permissions username@remote_host:/remote/path /local/mount/point

# Mount with compression
sshfs -o compression=yes username@remote_host:/remote/path /local/mount/point

# Mount with specific port
sshfs -o port=2222 username@remote_host:/remote/path /local/mount/point

# Unmount
fusermount -u /local/mount/point
```

---

## SSH Security Best Practices

### Key Security

```bash
# Use strong key types (Ed25519 recommended)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Use passphrases on private keys
ssh-keygen -t ed25519

# Set proper permissions
chmod 700 ~/.ssh/
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 644 ~/.ssh/authorized_keys

# Rotate keys regularly
# Generate new keys and update authorized_keys
```

### Server Security

```bash
# Disable root login
PermitRootLogin no

# Use key-based authentication only
PasswordAuthentication no

# Change default port
Port 2222

# Limit login attempts
MaxAuthTries 3
LoginGraceTime 30

# Use specific users
AllowUsers username1 username2

# Disable unused features
X11Forwarding no
AllowAgentForwarding no
AllowTcpForwarding no

# Use strong ciphers
Ciphers aes256-ctr,aes192-ctr,aes128-ctr
MACs hmac-sha2-256,hmac-sha2-512
KexAlgorithms diffie-hellman-group14-sha256,diffie-hellman-group16-sha512
```

### Network Security

```bash
# Use firewall rules
sudo ufw allow 22/tcp
sudo ufw allow from 192.168.1.0/24 to any port 22

# Use fail2ban
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban

# Monitor SSH logs
sudo tail -f /var/log/auth.log | grep sshd
```

### Client Security

```bash
# Use SSH config for consistent settings
Host production
    HostName prod.example.com
    User deploy
    IdentityFile ~/.ssh/id_ed25519_prod
    Port 2222
    ServerAliveInterval 60
    ServerAliveCountMax 3

# Verify host keys
# Check ~/.ssh/known_hosts for expected fingerprints

# Use SSH agent for key management
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

---

## SSH Troubleshooting

### Connection Issues

```bash
# Test connectivity
ping remote_host
telnet remote_host 22

# Check SSH service status
sudo systemctl status ssh

# Check SSH logs
sudo journalctl -u ssh
sudo tail -f /var/log/auth.log

# Test SSH connection with verbose output
ssh -vvv username@remote_host

# Check SSH configuration
sudo sshd -t

# Check SSH daemon configuration
sudo sshd -T | grep -E "(port|listenaddress|protocol)"
```

### Authentication Issues

```bash
# Check key permissions
ls -la ~/.ssh/
ls -la ~/.ssh/authorized_keys

# Verify public key on server
cat ~/.ssh/authorized_keys

# Test key authentication
ssh -i ~/.ssh/id_ed25519 username@remote_host

# Check SSH agent
ssh-add -l

# Regenerate host keys (if needed)
sudo rm /etc/ssh/ssh_host_*
sudo dpkg-reconfigure openssh-server
```

### Performance Issues

```bash
# Enable compression
ssh -C username@remote_host

# Use faster ciphers
ssh -c aes128-ctr username@remote_host

# Use faster MAC algorithms
ssh -m hmac-sha1 username@remote_host

# Optimize connection settings
ssh -o Compression=yes -o TCPKeepAlive=yes username@remote_host
```

### Common Error Messages

```bash
# "Permission denied (publickey)"
# - Check if public key is in authorized_keys
# - Check key permissions
# - Verify key type is accepted

# "Host key verification failed"
# - Remove old host key: ssh-keygen -R hostname
# - Accept new host key when prompted

# "Connection timed out"
# - Check firewall settings
# - Verify port is correct
# - Check network connectivity

# "Too many authentication failures"
# - Check MaxAuthTries setting
# - Verify key permissions
# - Check SSH agent
```

---

## Advanced SSH Features

### SSH Multiplexing

```bash
# Enable connection sharing
ssh -o ControlMaster=auto -o ControlPath=~/.ssh/control-%h-%p-%r username@remote_host

# Use existing connection
ssh -o ControlMaster=no -o ControlPath=~/.ssh/control-%h-%p-%r username@remote_host

# Configure in SSH config
Host *
    ControlMaster auto
    ControlPath ~/.ssh/control-%h-%p-%r
    ControlPersist 10m
```

### SSH Config with Advanced Options

```bash
# Advanced host configuration
Host production
    HostName prod.example.com
    User deploy
    Port 2222
    IdentityFile ~/.ssh/id_ed25519_prod
    ForwardAgent yes
    ForwardX11 no
    Compression yes
    ServerAliveInterval 60
    ServerAliveCountMax 3
    TCPKeepAlive yes
    ConnectTimeout 10
    BatchMode no
    StrictHostKeyChecking yes
    UserKnownHostsFile ~/.ssh/known_hosts_prod
    LogLevel INFO
```

### SSH Scripts and Automation

```bash
#!/bin/bash
# SSH backup script example
HOST="backup.example.com"
USER="backup"
KEY="~/.ssh/id_ed25519_backup"
SOURCE="/data"
DEST="/backup"

rsync -avz -e "ssh -i $KEY" $USER@$HOST:$SOURCE $DEST
```

### SSH with Custom Commands

```bash
# Execute remote command and capture output
result=$(ssh username@remote_host "echo 'Hello World'")

# Execute multiple commands
ssh username@remote_host << 'EOF'
cd /tmp
ls -la
pwd
whoami
EOF

# Execute with environment variables
ssh username@remote_host "export PATH=/usr/local/bin:\$PATH && command"
```

---

## Quick Reference

### Essential Commands

```bash
# Connect to remote host
ssh username@remote_host

# Generate SSH key
ssh-keygen -t ed25519

# Copy key to remote host
ssh-copy-id username@remote_host

# Copy file to remote host
scp file.txt username@remote_host:/path/

# Sync directory to remote host
rsync -av local/ username@remote_host:/remote/

# Create tunnel
ssh -L 8080:localhost:80 username@remote_host

# Mount remote directory
sshfs username@remote_host:/remote /local
```

### Common Options

```bash
-p PORT          # Specify port
-i KEYFILE       # Specify identity file
-v               # Verbose output
-C               # Enable compression
-f               # Background execution
-N               # Don't execute remote command
-L PORT:HOST:PORT # Local port forwarding
-R PORT:HOST:PORT # Remote port forwarding
-D PORT          # Dynamic port forwarding
```

### Configuration Files

```bash
/etc/ssh/sshd_config    # Server configuration
/etc/ssh/ssh_config     # Global client configuration
~/.ssh/config          # User client configuration
~/.ssh/authorized_keys  # Authorized public keys
~/.ssh/known_hosts     # Known host keys
~/.ssh/id_*           # Private keys
~/.ssh/id_*.pub       # Public keys
```

This comprehensive SSH guide covers all essential aspects of SSH usage on Linux systems. Remember to always follow security best practices and keep your SSH keys and configurations secure.
