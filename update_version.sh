#!/bin/bash

# Get the current version from pom.xml
CURRENT_VERSION=$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout)

# Split version into major, minor, and patch
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"

# Increment patch version
NEW_PATCH=$((PATCH + 1))
NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"

# Update version in pom.xml using mvn versions:set
mvn versions:set -DnewVersion=$NEW_VERSION -DgenerateBackupPoms=false

echo $NEW_VERSION