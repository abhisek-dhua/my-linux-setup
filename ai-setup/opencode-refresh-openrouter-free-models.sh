#!/bin/bash
# Update opencode.json with free OpenRouter models only
# Maintains config structure, only updates the models list

# Check if opencode is installed
if ! command -v opencode &> /dev/null; then
  echo "Error: opencode is not installed or not found in PATH"
  echo "Install it from: https://opencode.ai/docs/intro"
  exit 1
fi

echo "opencode found: $(command -v opencode)"

CONFIG_FILE="$HOME/.config/opencode/opencode.json"
CONFIG_DIR=$(dirname "$CONFIG_FILE")
TEMP_FILE=$(mktemp)

# Create config directory if it doesn't exist
if [ ! -d "$CONFIG_DIR" ]; then
  mkdir -p "$CONFIG_DIR"
  echo "Created directory: $CONFIG_DIR"
fi

# Fetch free models from OpenRouter API
echo "Fetching free OpenRouter models..."

curl -s "https://openrouter.ai/api/v1/models" | \
  jq -r '.data[] | select(.id | endswith(":free")) | .id' | \
  sort > "$TEMP_FILE"

if [ ! -s "$TEMP_FILE" ]; then
  echo "Error: Failed to fetch models from OpenRouter"
  rm "$TEMP_FILE"
  exit 1
fi

# Generate the models JSON object
MODELS_JSON=$(cat "$TEMP_FILE" | awk '
BEGIN { print "      \"models\": {" }
NR>1 { print "," }
{ printf "        \"%s\": {}", $0 }
END { print "      }" }
')

# Create updated config with free models
cat > "$CONFIG_FILE" << 'EOF'
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "openrouter": {
EOF

# Append the dynamically generated models
echo "$MODELS_JSON" >> "$CONFIG_FILE"

# Close the JSON structure
cat >> "$CONFIG_FILE" << 'EOF'
    }
  }
}
EOF

rm "$TEMP_FILE"
echo "Updated $CONFIG_FILE with $(grep -c ':free' "$CONFIG_FILE") free models"
