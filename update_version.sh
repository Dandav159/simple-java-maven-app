#!/bin/bash

current_version=$(mvn help:evaluate -Dexpression=project.version -q -DforceStdout)

IFS='.' read -r major minor patch <<< "$current_version"
patch=$((patch + 1))
new_version="$major.$minor.$patch"

mvn versions:set -DnewVersion="$new_version"