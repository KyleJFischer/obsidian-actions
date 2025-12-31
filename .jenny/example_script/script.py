#!/usr/bin/env python3
# Example script that receives note data as JSON file

import json
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Error: No JSON file provided")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"Error: JSON file not found: {json_file}")
        sys.exit(1)
    
    print(f"Processing note from: {json_file}")
    print(f"JSON file: {json_file}")
    
    # Read and parse JSON file
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        sys.exit(1)
    
    # Print the raw JSON
    print(json.dumps(data, indent=2))
    
    # Extract and display note information
    title = data.get('title', '')
    path = data.get('path', '')
    tags = data.get('tags', [])
    properties = data.get('properties', {})
    
    print(f"Title: {title}")
    print(f"Path: {path}")
    print(f"Tags: {tags}")
    print(f"Properties: {properties}")
    
    # Example: You could do something with the note here
    # For instance, send it to an API, update a database, etc.
    
    print("Script completed successfully")

if __name__ == "__main__":
    main()
