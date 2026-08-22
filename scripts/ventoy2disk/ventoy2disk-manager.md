# Ventoy Setup Guide for Pendrive

**Latest version:** 1.1.15 (released 2026-06-25)

---

## Prerequisites

- Linux system
- `curl`, `sudo`
- Pendrive (e.g., `/dev/sda` -- **all data will be erased**)

Install required tools:

```bash
sudo apt install -y exfatprogs
sudo ln -sf /usr/sbin/mkfs.exfat /usr/sbin/mkexfatfs
```

## Download & Extract

```bash
curl -L -o /tmp/ventoy.tar.gz \
  "https://github.com/ventoy/Ventoy/releases/download/v1.1.15/ventoy-1.1.15-linux.tar.gz"
tar -xzf /tmp/ventoy.tar.gz -C /tmp
cd /tmp/ventoy-1.1.15
```

## Fresh Installation

### MBR (default)

```bash
sudo sh Ventoy2Disk.sh -i /dev/sdX
```

### GPT

```bash
sudo sh Ventoy2Disk.sh -i -g /dev/sdX
```

To force reinstall over an existing Ventoy installation, use `-I` instead of `-i`.

## Version Upgrade

To update Ventoy on the pendrive without losing existing ISO files:

```bash
sudo sh Ventoy2Disk.sh -u /dev/sdX
```

## Verify Installation

```bash
sudo sh Ventoy2Disk.sh -l /dev/sdX
```

Output shows version, partition style (MBR/GPT), and Secure Boot status.

## Notes

- The `-u` flag updates Ventoy safely without touching the data partition.
- The `-i` and `-I` flags **erase all data** on the drive.
- `-g` sets GPT partition style (omit for MBR).
- `-s` / `-S` enable/disable Secure Boot support.
