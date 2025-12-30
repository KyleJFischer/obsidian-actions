#!/bin/bash
# Example script that receives note data as JSON file

JSON_FILE="$1"

if [ -z "$JSON_FILE" ]; then
    echo "Error: No JSON file provided"
    exit 1
fi

echo "Processing note from: $JSON_FILE"

# Read and display note information
TITLE=$(jq -r '.title' "$JSON_FILE")
PATH=$(jq -r '.path' "$JSON_FILE")
TAGS=$(jq -r '.tags | join(", ")' "$JSON_FILE")

echo "Title: $TITLE"
echo "Path: $PATH"
echo "Tags: $TAGS"

# Example: You could do something with the note here
# For instance, send it to an API, update a database, etc.

echo "Script completed successfully"
