#!/bin/bash

version=$(grep -oP "(?<=__version__ = ['\"])[^'\"]+" mlfcrafter/__init__.py)

if [ -z "$version" ]; then
  echo "Version not found"
  exit 1
else
  echo "$version"
fi