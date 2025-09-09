#!/bin/bash
# Sync version number from CMakeLists.txt to all other files
# Usage: ./scripts/sync_version.sh

set -e

# Get version components from CMakeLists.txt
MAJOR=$(grep 'set(PROJECT_VERSION_MAJOR' CMakeLists.txt | sed 's/.*"\([0-9]*\)".*/\1/')
MINOR=$(grep 'set(PROJECT_VERSION_MINOR' CMakeLists.txt | sed 's/.*"\([0-9]*\)".*/\1/')
PATCH=$(grep 'set(PROJECT_VERSION_PATCH' CMakeLists.txt | sed 's/.*"\([0-9]*\)".*/\1/')
VERSION="$MAJOR.$MINOR.$PATCH"

echo "Syncing version: $VERSION"

# Update package.json
if [ -f "packages/occjs/package.json" ]; then
    sed -i.bak 's/"version": "[^"]*"/"version": "'"$VERSION"'"/' packages/occjs/package.json
    rm packages/occjs/package.json.bak
    echo "✓ Updated packages/occjs/package.json"
fi

# Update pyproject.toml
if [ -f "pyproject.toml" ]; then
    sed -i.bak 's/^version = "[^"]*"/version = "'"$VERSION"'"/' pyproject.toml
    rm pyproject.toml.bak
    echo "✓ Updated pyproject.toml"
fi

# Update occpy.cpp fallback version
if [ -f "src/occpy.cpp" ]; then
    sed -i.bak 's/m\.attr("__version__") = "[^"]*";/m.attr("__version__") = "'"$VERSION"'";/' src/occpy.cpp
    echo "✓ Updated src/occpy.cpp fallback version"
    rm src/occpy.cpp.bak
fi

echo "Version sync complete: $VERSION"