# Jenny

A GitHub Action that runs scripts based on note changes and filters. Jenny detects changed markdown files in git pushes, parses them into a structured format, and executes scripts when notes match filter criteria.

## Features

- **Automatic Detection**: Detects changed markdown files from git push events
- **Note Parsing**: Extracts title, path, content, tags, and properties (frontmatter) from markdown files
- **Flexible Filtering**: Filter notes by path patterns, tags, properties, and change types
- **Script Execution**: Execute any script (bash, Python, etc.) when notes match filters
- **JSON Data Passing**: Notes are passed to scripts as JSON files for easy processing

## How It Works

1. **Git Push Event**: Jenny triggers on git push events
2. **Change Detection**: Detects all changed markdown files (`.md`, `.markdown`)
3. **Note Parsing**: Parses each changed file to extract:
   - `title`: From frontmatter `title` field or first H1 heading
   - `path`: Relative file path
   - `content`: Full markdown content
   - `tags`: From frontmatter `tags` array and inline `#tag` patterns
   - `properties`: YAML frontmatter as a dictionary
4. **Filter Evaluation**: For each script in `.jenny/`, checks if notes match the filter criteria
5. **Script Execution**: When a note matches, writes note data as JSON to a temp file and executes the script

## Installation

### As a GitHub Action

Add Jenny to your notes repository's workflow:

```yaml
name: Process Notes

on:
  push:
    branches:
      - main

jobs:
  jenny:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - uses: your-org/jenny@main
        with:
          jenny_dir: '.jenny'
```

### Local Development

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r action/requirements.txt
   ```
3. Run the runner:
   ```bash
   python action/jenny_runner.py .jenny
   ```

## Configuration

### Directory Structure

Create a `.jenny` directory in your notes repository root. Each subdirectory represents a filter-to-script mapping:

```
.jenny/
├── script1/
│   ├── filter.yaml
│   └── script.sh
├── script2/
│   ├── filter.yaml
│   └── script.py
└── ...
```

### Filter Configuration

Each script directory must contain a `filter.yaml` file:

```yaml
name: My Filter
description: Description of what this filter does
enabled: true

filters:
  - type: path
    params:
      include: ["**/*.md"]
      exclude: ["drafts/**"]
  
  - type: tag
    params:
      has_any: ["important", "publish"]
  
  - type: property
    params:
      key: "status"
      value: "published"
  
  - type: change_type
    params:
      types: ["added", "modified"]

script: "./script.sh"  # Relative to filter.yaml directory
```

### Filter Types

#### Path Filter

Filter notes by file path patterns (glob syntax):

```yaml
- type: path
  params:
    include: ["notes/**/*.md", "posts/*.md"]
    exclude: ["drafts/**", "archive/**"]
```

#### Tag Filter

Filter notes by tags:

```yaml
- type: tag
  params:
    has_any: ["important", "publish"]  # Note must have at least one
    has_all: ["review", "approved"]     # Note must have all
    has_none: ["draft", "private"]     # Note must have none
```

#### Property Filter

Filter notes by frontmatter properties:

```yaml
- type: property
  params:
    key: "status"
    value: "published"  # Exact match
    # OR
    exists: true        # Property exists (any value)
```

#### Change Type Filter

Filter notes by git change type:

```yaml
- type: change_type
  params:
    types: ["added", "modified"]  # Options: added, modified, deleted, renamed
```

#### Composite Filter

Combine multiple filters with AND/OR logic:

```yaml
- type: composite
  params:
    operator: and  # or "or"
    filters:
      - type: tag
        params:
          has_any: ["publish"]
      - type: property
        params:
          key: "status"
          value: "ready"
```

### Script Configuration

The `script` field in `filter.yaml` specifies the script to execute. It should be relative to the filter directory:

```yaml
script: "./script.sh"     # Bash script
script: "./script.py"     # Python script
script: "./custom_script" # Any executable
```

Scripts must be executable and will receive the JSON file path as their first argument.

## Note JSON Format

Scripts receive note data as a JSON file. The JSON structure is:

```json
{
  "title": "Note Title",
  "path": "notes/example.md",
  "content": "# Note Title\n\nContent here...",
  "tags": ["tag1", "tag2"],
  "properties": {
    "status": "published",
    "date": "2024-01-01",
    "author": "John Doe"
  },
  "change_type": "added",
  "commit_sha": "abc123def456...",
  "commit_message": "Add new note"
}
```

## Example Scripts

### Bash Script

```bash
#!/bin/bash
JSON_FILE="$1"

TITLE=$(jq -r '.title' "$JSON_FILE")
PATH=$(jq -r '.path' "$JSON_FILE")
TAGS=$(jq -r '.tags | join(", ")' "$JSON_FILE")

echo "Processing: $TITLE"
echo "Path: $PATH"
echo "Tags: $TAGS"

# Your processing logic here
```

### Python Script

```python
#!/usr/bin/env python3
import json
import sys

json_file = sys.argv[1]

with open(json_file, 'r') as f:
    note = json.load(f)

print(f"Processing: {note['title']}")
print(f"Path: {note['path']}")
print(f"Tags: {', '.join(note['tags'])}")

# Your processing logic here
```

## Note Format

Jenny expects markdown files with optional YAML frontmatter:

```markdown
---
title: My Note Title
status: published
tags:
  - important
  - publish
date: 2024-01-01
---

# My Note Title

This is the content of my note. I can also use inline #tags here.
```

- **Title**: Extracted from frontmatter `title` field, or first H1 heading, or filename
- **Tags**: Extracted from frontmatter `tags` array and inline `#tag` patterns
- **Properties**: All frontmatter fields become properties
- **Content**: Full markdown content including frontmatter

## Environment Variables

Jenny uses the following GitHub Actions environment variables:

- `GITHUB_SHA`: The commit SHA that triggered the workflow
- `GITHUB_BEFORE`: The commit SHA before the push (for detecting changes)

These are automatically available in GitHub Actions.

## Error Handling

- If a note fails to parse, it's logged and skipped
- If a filter config fails to load, it's logged and skipped
- If a script execution fails, it's logged but doesn't stop other scripts
- The action exits with code 1 if any script executions fail

## Limitations

- Only processes markdown files (`.md`, `.markdown`)
- Deleted files have minimal metadata (no content)
- Scripts must be executable
- Scripts run sequentially (not in parallel)

## License

[Add your license here]
