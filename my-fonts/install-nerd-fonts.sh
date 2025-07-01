#!/bin/bash

# Target fonts directory
FONT_DIR="$HOME/.fonts"
mkdir -p "$FONT_DIR"

# Nerd fonts to download
declare -a fonts=(
  "JetBrainsMono"
  "FiraCode"
  "Hack"
  "CascadiaCode"
  "Meslo"
  "UbuntuMono"
)

# Base URL for latest Nerd Fonts
BASE_URL="https://github.com/ryanoasis/nerd-fonts/releases/latest/download"

echo "🔽 Downloading Nerd Fonts..."

for font in "${fonts[@]}"; do
  ZIP_NAME="${font}.zip"
  URL="${BASE_URL}/${ZIP_NAME}"
  DEST="${FONT_DIR}/${font}"
  mkdir -p "$DEST"

  echo "📦 Downloading $font..."
  wget -q --show-progress "$URL" -O "/tmp/${ZIP_NAME}"

  echo "📂 Extracting $font to $DEST..."
  unzip -o "/tmp/${ZIP_NAME}" -d "$DEST"

  echo "✅ $font installed."
done

# Refresh font cache
echo "🔄 Updating font cache..."
fc-cache -fv

echo "🎉 All Nerd Fonts installed successfully!"
