#!/bin/bash

# Directory to store downloaded zips and extracted fonts
DOWNLOAD_DIR="./nerd-fonts-zips"
EXTRACT_DIR="./nerd-fonts"
mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$EXTRACT_DIR"

# Nerd fonts to download and extract
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

echo "🔽 Downloading and extracting Nerd Fonts locally..."

for font in "${fonts[@]}"; do
  ZIP_NAME="${font}.zip"
  URL="${BASE_URL}/${ZIP_NAME}"
  ZIP_PATH="${DOWNLOAD_DIR}/${ZIP_NAME}"
  FONT_OUTPUT="${EXTRACT_DIR}/${font}"

  echo "📦 Downloading $font..."
  wget -q --show-progress "$URL" -O "$ZIP_PATH"

  echo "📂 Extracting to $FONT_OUTPUT..."
  mkdir -p "$FONT_OUTPUT"
  unzip -o "$ZIP_PATH" -d "$FONT_OUTPUT"

  echo "✅ $font extracted."
done

echo "🎉 All Nerd Fonts downloaded and extracted to '$EXTRACT_DIR'!"

