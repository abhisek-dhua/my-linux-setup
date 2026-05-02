# Ultimate Linux File System Guide

## Table of Contents

1. [Basic File Operations](#basic-file-operations)
2. [File Viewing and Navigation](#file-viewing-and-navigation)
3. [File Search and Discovery](#file-search-and-discovery)
4. [Hidden Files and Directories](#hidden-files-and-directories)
5. [File Permissions and Ownership](#file-permissions-and-ownership)
6. [Remote File Access](#remote-file-access)
7. [Advanced File Operations](#advanced-file-operations)
8. [File System Monitoring](#file-system-monitoring)

---

## Basic File Operations

### Copying Files and Directories

#### `cp` Command

```bash
# Basic file copy
cp source_file destination_file

# Copy with verbose output
cp -v source_file destination_file

# Copy and preserve attributes (permissions, timestamps)
cp -p source_file destination_file

# Copy recursively (for directories)
cp -r source_directory destination_directory

# Copy with confirmation prompts
cp -i source_file destination_file

# Copy multiple files to a directory
cp file1 file2 file3 destination_directory/

# Copy with progress bar (if available)
cp --progress source_file destination_file
```

#### Examples:

```bash
# Copy a file to current directory
cp /path/to/source.txt ./destination.txt

# Copy to home directory
cp file.txt ~/

# Copy with backup (adds ~ suffix)
cp -b file.txt backup/

# Copy preserving all attributes
cp -a source_directory/ destination_directory/
```

### Moving and Renaming Files

#### `mv` Command

```bash
# Move/rename a file
mv old_name new_name

# Move file to directory
mv file.txt /path/to/directory/

# Move with verbose output
mv -v source destination

# Move with confirmation
mv -i source destination

# Move with backup
mv -b source destination

# Move multiple files
mv file1 file2 file3 destination_directory/
```

#### Examples:

```bash
# Rename a file
mv old_filename.txt new_filename.txt

# Move to parent directory
mv file.txt ../

# Move with backup
mv -b important_file.txt backup/
```

### Creating Files and Directories

#### `touch` Command

```bash
# Create empty file
touch filename.txt

# Create multiple files
touch file1.txt file2.txt file3.txt

# Update access time only
touch -a filename.txt

# Update modification time only
touch -m filename.txt

# Set specific timestamp
touch -t 202312011200 filename.txt
```

#### `mkdir` Command

```bash
# Create directory
mkdir directory_name

# Create nested directories
mkdir -p parent/child/grandchild

# Create with specific permissions
mkdir -m 755 directory_name

# Create multiple directories
mkdir dir1 dir2 dir3
```

---

## File Viewing and Navigation

### Viewing File Contents

#### `cat` Command

```bash
# Display file contents
cat filename.txt

# Display with line numbers
cat -n filename.txt

# Display with non-printing characters
cat -A filename.txt

# Concatenate multiple files
cat file1.txt file2.txt > combined.txt

# Display with end-of-line markers
cat -E filename.txt
```

#### `less` Command

```bash
# View file with pagination
less filename.txt

# Search within less
# Press / to search forward, ? to search backward

# Navigation in less:
# Space: Next page
# b: Previous page
# g: Go to beginning
# G: Go to end
# q: Quit
```

#### `head` and `tail` Commands

```bash
# View first 10 lines
head filename.txt

# View first N lines
head -n 20 filename.txt

# View last 10 lines
tail filename.txt

# View last N lines
tail -n 20 filename.txt

# Follow file in real-time
tail -f logfile.txt

# Follow with multiple files
tail -f file1.txt file2.txt
```

### Directory Navigation

#### `ls` Command

```bash
# List files and directories
ls

# List with details
ls -l

# List all files (including hidden)
ls -a

# List with human-readable sizes
ls -lh

# List sorted by modification time
ls -lt

# List sorted by size
ls -lS

# List with file types
ls -F

# List with colors
ls --color=auto

# List recursively
ls -R
```

#### `pwd` Command

```bash
# Print working directory
pwd

# Print working directory (logical)
pwd -L

# Print working directory (physical)
pwd -P
```

#### `cd` Command

```bash
# Change to home directory
cd
cd ~

# Change to parent directory
cd ..

# Change to specific directory
cd /path/to/directory

# Change to previous directory
cd -

# Change to home directory of user
cd ~username
```

---

## File Search and Discovery

### Finding Files

#### `find` Command

```bash
# Find files by name
find /path/to/search -name "filename.txt"

# Find files by pattern
find /path/to/search -name "*.txt"

# Find files by type
find /path/to/search -type f  # files only
find /path/to/search -type d  # directories only

# Find files by size
find /path/to/search -size +100M  # larger than 100MB
find /path/to/search -size -1M    # smaller than 1MB

# Find files by modification time
find /path/to/search -mtime -7    # modified in last 7 days
find /path/to/search -mtime +30   # modified more than 30 days ago

# Find files by permissions
find /path/to/search -perm 644

# Find files by owner
find /path/to/search -user username

# Find files by group
find /path/to/search -group groupname

# Execute command on found files
find /path/to/search -name "*.tmp" -delete
find /path/to/search -name "*.log" -exec cp {} backup/ \;
```

#### `locate` Command

```bash
# Find files using database
locate filename.txt

# Update locate database
sudo updatedb

# Case-insensitive search
locate -i filename.txt

# Limit results
locate -n 10 filename.txt
```

#### `which` and `whereis` Commands

```bash
# Find executable in PATH
which command_name

# Find binary, source, and manual pages
whereis command_name
```

---

## Hidden Files and Directories

### Working with Hidden Files

#### Viewing Hidden Files

```bash
# List all files including hidden
ls -a

# List only hidden files
ls -d .*

# List hidden files with details
ls -la | grep "^\."

# Show hidden files in current directory only
ls -d .[^.]*
```

#### Creating Hidden Files

```bash
# Create hidden file
touch .hidden_file

# Create hidden directory
mkdir .hidden_directory

# Copy file as hidden
cp file.txt .hidden_file.txt
```

#### Common Hidden Files and Directories

```bash
# User configuration files
~/.bashrc
~/.bash_profile
~/.profile
~/.ssh/
~/.config/

# System configuration
/etc/
/var/log/

# Application data
~/.cache/
~/.local/
```

---

## File Permissions and Ownership

### Understanding Permissions

#### `chmod` Command

```bash
# Change permissions using octal notation
chmod 755 filename
chmod 644 filename
chmod 600 filename

# Change permissions using symbolic notation
chmod u+x filename      # Add execute for user
chmod g-w filename      # Remove write for group
chmod o=r filename      # Set read-only for others
chmod a+x filename      # Add execute for all

# Change permissions recursively
chmod -R 755 directory/

# Change permissions with reference
chmod --reference=source_file target_file
```

#### `chown` Command

```bash
# Change owner
chown username filename

# Change owner and group
chown username:groupname filename

# Change recursively
chown -R username:groupname directory/

# Change with reference
chown --reference=source_file target_file
```

#### `chgrp` Command

```bash
# Change group
chgrp groupname filename

# Change group recursively
chgrp -R groupname directory/
```

---

## Remote File Access

### SSH File Transfer

#### `scp` Command (Secure Copy)

```bash
# Copy file from local to remote
scp local_file.txt user@remote_host:/path/to/destination/

# Copy file from remote to local
scp user@remote_host:/path/to/remote_file.txt ./

# Copy directory recursively
scp -r local_directory/ user@remote_host:/path/to/destination/

# Copy with specific port
scp -P 2222 local_file.txt user@remote_host:/path/to/destination/

# Copy with compression
scp -C local_file.txt user@remote_host:/path/to/destination/

# Copy with verbose output
scp -v local_file.txt user@remote_host:/path/to/destination/

# Copy multiple files
scp file1.txt file2.txt user@remote_host:/path/to/destination/
```

#### `rsync` Command

```bash
# Sync files (one-way)
rsync -av local_directory/ user@remote_host:/path/to/destination/

# Sync with compression
rsync -avz local_directory/ user@remote_host:/path/to/destination/

# Sync with progress
rsync -av --progress local_directory/ user@remote_host:/path/to/destination/

# Sync with exclude patterns
rsync -av --exclude='*.tmp' --exclude='*.log' local_directory/ user@remote_host:/path/to/destination/

# Sync with delete (mirror)
rsync -av --delete local_directory/ user@remote_host:/path/to/destination/

# Sync with bandwidth limit
rsync -av --bwlimit=1000 local_directory/ user@remote_host:/path/to/destination/

# Dry run (simulate)
rsync -av --dry-run local_directory/ user@remote_host:/path/to/destination/
```

#### `sftp` Command

```bash
# Connect to remote host
sftp user@remote_host

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

### Mounting Remote File Systems

#### SSHFS (SSH File System)

```bash
# Install SSHFS
sudo apt install sshfs  # Ubuntu/Debian
sudo yum install fuse-sshfs  # CentOS/RHEL

# Mount remote directory
sshfs user@remote_host:/remote/path /local/mount/point

# Mount with specific options
sshfs -o allow_other,default_permissions user@remote_host:/remote/path /local/mount/point

# Mount with compression
sshfs -o compression=yes user@remote_host:/remote/path /local/mount/point

# Unmount
fusermount -u /local/mount/point
```

---

## Advanced File Operations

### File Compression and Archiving

#### `tar` Command

```bash
# Create archive
tar -cf archive.tar files/

# Create compressed archive
tar -czf archive.tar.gz files/
tar -cjf archive.tar.bz2 files/
tar -cJf archive.tar.xz files/

# Extract archive
tar -xf archive.tar
tar -xzf archive.tar.gz
tar -xjf archive.tar.bz2
tar -xJf archive.tar.xz

# List archive contents
tar -tf archive.tar

# Extract to specific directory
tar -xzf archive.tar.gz -C /path/to/extract/

# Create archive with verbose output
tar -czvf archive.tar.gz files/
```

#### `zip` and `unzip` Commands

```bash
# Create zip archive
zip archive.zip files/

# Create password-protected archive
zip -P password archive.zip files/

# Extract zip archive
unzip archive.zip

# Extract to specific directory
unzip archive.zip -d /path/to/extract/

# List zip contents
unzip -l archive.zip
```

### File Comparison

#### `diff` Command

```bash
# Compare two files
diff file1.txt file2.txt

# Compare with context
diff -c file1.txt file2.txt

# Compare with unified format
diff -u file1.txt file2.txt

# Compare directories
diff -r directory1/ directory2/

# Ignore whitespace
diff -w file1.txt file2.txt

# Ignore case
diff -i file1.txt file2.txt
```

#### `cmp` Command

```bash
# Compare files byte by byte
cmp file1.txt file2.txt

# Show first difference
cmp -l file1.txt file2.txt
```

### File Linking

#### `ln` Command

```bash
# Create symbolic link
ln -s target_file link_name

# Create hard link
ln target_file link_name

# Create link with absolute path
ln -s /absolute/path/to/target link_name

# Remove link
rm link_name

# Update existing link
ln -sf new_target existing_link
```

---

## File System Monitoring

### Disk Usage

#### `df` Command

```bash
# Show disk usage
df

# Show disk usage in human-readable format
df -h

# Show disk usage for specific filesystem
df -h /home

# Show inode usage
df -i
```

#### `du` Command

```bash
# Show directory size
du directory/

# Show size in human-readable format
du -h directory/

# Show total size only
du -sh directory/

# Show sizes for all subdirectories
du -h --max-depth=1 directory/

# Exclude certain patterns
du -h --exclude='*.tmp' directory/
```

### File System Monitoring Tools

#### `inotify` Tools

```bash
# Monitor file changes
inotifywait -m directory/

# Monitor specific events
inotifywait -m -e modify,create,delete directory/

# Monitor recursively
inotifywait -m -r directory/
```

#### `fswatch` Command

```bash
# Monitor file system events
fswatch directory/

# Monitor with specific events
fswatch -o directory/ | xargs -n1 -I{} echo "Changed: {}"
```

---

## Best Practices

### File Management

1. **Use meaningful names**: Choose descriptive file and directory names
2. **Organize hierarchically**: Use logical directory structures
3. **Backup regularly**: Keep backups of important files
4. **Use version control**: For code and documents
5. **Clean up regularly**: Remove temporary and unnecessary files

### Security

1. **Set appropriate permissions**: Don't make files world-writable
2. **Use SSH keys**: For remote access instead of passwords
3. **Encrypt sensitive data**: Use encryption for confidential files
4. **Regular updates**: Keep system and tools updated

### Performance

1. **Use appropriate tools**: Choose the right command for the job
2. **Batch operations**: Use loops and scripts for repetitive tasks
3. **Monitor disk usage**: Prevent disk space issues
4. **Use compression**: For large files and archives

---

## Quick Reference

### Essential Commands

```bash
# File operations
cp source dest          # Copy
mv source dest          # Move/rename
rm file                 # Remove
mkdir dir               # Create directory
rmdir dir               # Remove directory

# Viewing
cat file                # View file
less file               # View with pagination
head file               # View first lines
tail file               # View last lines
ls                      # List files

# Searching
find path -name "file"  # Find files
grep "pattern" file     # Search in file
locate file             # Find using database

# Permissions
chmod 755 file          # Change permissions
chown user file         # Change owner
chgrp group file        # Change group

# Remote
scp file user@host:/path    # Copy to remote
rsync -av src/ user@host:/dest/  # Sync to remote
sshfs user@host:/path /local     # Mount remote
```

This guide covers the essential Linux file system operations. Remember to always be careful with destructive operations and test commands on non-critical files first.
