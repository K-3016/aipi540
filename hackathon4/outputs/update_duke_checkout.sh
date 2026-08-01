#!/bin/zsh
set -euo pipefail

SOURCE_REPO="/Users/kai/Documents/Codex/2026-07-26/you-are-an-expert-machine-learning"
TARGET_REPO="/Users/kai/Documents/Duke_AIPI/aipi540/aipi540"
ADAPTER_DIR="$TARGET_REPO/models/medexplain_lora_adapter"

if [[ ! -d "$TARGET_REPO/.git" ]]; then
  print -u2 "Expected Git checkout not found: $TARGET_REPO"
  exit 1
fi

mkdir -p "$ADAPTER_DIR"
cp "$SOURCE_REPO/.gitignore" "$TARGET_REPO/.gitignore"
cp "$SOURCE_REPO/README.md" "$TARGET_REPO/README.md"
cp "$SOURCE_REPO/main.py" "$TARGET_REPO/main.py"
cp "$SOURCE_REPO/scripts/model.py" "$TARGET_REPO/scripts/model.py"
cp "$SOURCE_REPO/models/medexplain_lora_adapter/adapter_config.json" \
  "$ADAPTER_DIR/adapter_config.json"
cp "$SOURCE_REPO/models/medexplain_lora_adapter/adapter_model.safetensors" \
  "$ADAPTER_DIR/adapter_model.safetensors"

print "Updated deployment files in: $TARGET_REPO"
print "Unrelated untracked files were not modified."
print
git -C "$TARGET_REPO" status --short --branch
