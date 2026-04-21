#!/usr/bin/env bash
# Download the outdoors Q&A dataset from the AI-Powered Search book's data repo
# and extract posts.csv to data/outdoors/.
#
# The tarball is split into 20 parts on GitHub because of the 100 MB file size
# limit. This script is a tiny replacement for `aips.indexer.download_data_files`
# so we don't have to install the whole aips package just to fetch the CSV.

set -euo pipefail

DEST="data/outdoors"
BASE_URL="https://github.com/ai-powered-search/outdoors/raw/master"

mkdir -p "$DEST"
cd "$DEST"

echo "Downloading 20 tarball parts..."
for p in a b c d e f g h i j k l m n o p q r s t; do
  curl -fsSL -o "outdoors.tgz.part_$p" "$BASE_URL/outdoors.tgz.part_$p"
done

echo "Concatenating and extracting..."
cat outdoors.tgz.part_* > outdoors.tgz
tar -xzf outdoors.tgz
rm outdoors.tgz outdoors.tgz.part_*

# Some tarballs wrap contents in an extra directory. Flatten if so.
if [ -f outdoors/posts.csv ] && [ ! -f posts.csv ]; then
  mv outdoors/* .
  rmdir outdoors
fi

echo "Done. posts.csv is at: $(pwd)/posts.csv"
ls -lh posts.csv
wc -l posts.csv
