#!/bin/bash
echo "=== Hidden files in project ==="
find . -type f -name ".*" -not -path "./.git/*" -not -name ".gitignore" -not -name ".dockerignore"

echo ""
echo "=== Files that will be sent to Docker ==="
find . -type f -not -path "./.git/*" | head -20