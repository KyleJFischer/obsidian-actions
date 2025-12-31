#!/usr/bin/env python3
"""
Obsidian Actions Runner - Executes scripts based on note changes and filters.
"""

import json
import logging
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from fnmatch import fnmatch

import frontmatter
import git
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Type of change detected in git"""
    ADDED = "added"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"


@dataclass
class Note:
    """Represents a parsed note"""
    title: str
    path: str
    content: str
    tags: List[str]
    properties: Dict[str, Any]
    change_type: ChangeType
    commit_sha: str
    commit_message: str


class MarkdownParser:
    """Parses markdown files to extract note metadata"""

    INLINE_TAG_PATTERN = re.compile(r'#([a-zA-Z0-9_/-]+)')
    TITLE_PATTERN = re.compile(r'^#\s+(.+)$', re.MULTILINE)

    def parse(self, content: str, file_path: Path) -> Dict[str, Any]:
        """
        Parse markdown content and extract metadata.

        Returns dict with:
        - title: From frontmatter or first H1
        - tags: From frontmatter and inline tags
        - properties: Frontmatter as dict
        """
        try:
            post = frontmatter.loads(content)
            fm_data = dict(post.metadata)
        except Exception as e:
            logger.warning(f"Error parsing frontmatter for {file_path}: {e}")
            fm_data = {}
            post = frontmatter.Post(content)

        # Extract tags
        tags = self._extract_tags(post.content, fm_data)
        tags_list = sorted(list(tags))
        logger.info(f"Extracted {len(tags_list)} tags from {file_path}: {tags_list}")
        if 'tags' in fm_data:
            logger.info(f"Frontmatter tags value: {fm_data['tags']} (type: {type(fm_data['tags'])})")

        # Extract title
        title = self._extract_title(post.content, fm_data, file_path)

        return {
            'title': title,
            'tags': tags_list,
            'properties': fm_data,
            'content': content
        }

    def _extract_tags(self, content: str, frontmatter: Dict[str, Any]) -> Set[str]:
        """Extract tags from both frontmatter and inline content"""
        tags = set()

        # Frontmatter tags
        if 'tags' in frontmatter:
            fm_tags = frontmatter['tags']
            if isinstance(fm_tags, str):
                tags.add(fm_tags.strip())
            elif isinstance(fm_tags, (list, tuple)):
                # Handle both lists and tuples
                for t in fm_tags:
                    if t is not None:
                        tag_str = str(t).strip()
                        if tag_str:
                            tags.add(tag_str)

        # Inline tags
        inline_tags = self.INLINE_TAG_PATTERN.findall(content)
        tags.update(inline_tags)

        return tags

    def _extract_title(self, content: str, frontmatter: Dict[str, Any], file_path: Path) -> str:
        """Extract title from frontmatter or first H1"""
        # Try frontmatter first
        if 'title' in frontmatter:
            return str(frontmatter['title'])

        # Try first H1
        match = self.TITLE_PATTERN.search(content)
        if match:
            return match.group(1).strip()

        # Fallback to filename
        return file_path.stem


class GitDetector:
    """Detects changed markdown files from git commits"""

    def __init__(self, repo_path: Path):
        self.repo = git.Repo(repo_path)

    def get_changed_files(self, before_sha: Optional[str] = None, after_sha: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all changed markdown files between two commits.

        Returns list of dicts with:
        - file_path: Path object
        - change_type: ChangeType enum
        - commit_sha: str
        - commit_message: str
        """
        if before_sha is None and after_sha is None:
            # Use HEAD and HEAD~1
            after_commit = self.repo.head.commit
            if after_commit.parents:
                before_commit = after_commit.parents[0]
            else:
                # First commit - all files are added
                return self._get_all_files_from_commit(after_commit)
        else:
            if after_sha is None:
                after_commit = self.repo.head.commit
            else:
                after_commit = self.repo.commit(after_sha)

            if before_sha is None:
                if after_commit.parents:
                    before_commit = after_commit.parents[0]
                else:
                    return self._get_all_files_from_commit(after_commit)
            else:
                before_commit = self.repo.commit(before_sha)

        changes = []
        diffs = before_commit.diff(after_commit)

        for diff_item in diffs:
            change_type = self._map_change_type(diff_item.change_type)
            file_path = Path(diff_item.b_path if diff_item.b_path else diff_item.a_path)

            # Only track markdown files
            if not self._is_markdown_file(file_path):
                continue

            changes.append({
                'file_path': file_path,
                'change_type': change_type,
                'commit_sha': after_commit.hexsha,
                'commit_message': after_commit.message.strip()
            })

        return changes

    def _get_all_files_from_commit(self, commit) -> List[Dict[str, Any]]:
        """Get all markdown files from a commit (for first commit)"""
        changes = []
        for item in commit.tree.traverse():
            if item.type == 'blob':
                file_path = Path(item.path)
                if self._is_markdown_file(file_path):
                    changes.append({
                        'file_path': file_path,
                        'change_type': ChangeType.ADDED,
                        'commit_sha': commit.hexsha,
                        'commit_message': commit.message.strip()
                    })
        return changes

    def get_file_content(self, file_path: Path, commit_sha: str) -> str:
        """Get file content at a specific commit"""
        commit = self.repo.commit(commit_sha)
        return commit.tree[str(file_path)].data_stream.read().decode('utf-8')

    def _map_change_type(self, git_change_type: str) -> ChangeType:
        """Map git change types to our enum"""
        mapping = {
            'A': ChangeType.ADDED,
            'M': ChangeType.MODIFIED,
            'D': ChangeType.DELETED,
            'R': ChangeType.RENAMED,
        }
        return mapping.get(git_change_type, ChangeType.MODIFIED)

    def _is_markdown_file(self, path: Path) -> bool:
        return path.suffix.lower() in ['.md', '.markdown']


class Filter:
    """Base class for filters"""

    def __init__(self, params: Dict[str, Any]):
        self.params = params

    def matches(self, note: Note) -> bool:
        """Return True if the note matches this filter"""
        raise NotImplementedError


class PathFilter(Filter):
    """Filter based on file path patterns"""

    def matches(self, note: Note) -> bool:
        include_patterns = self.params.get('include', [])
        exclude_patterns = self.params.get('exclude', [])

        # Check excludes first
        for pattern in exclude_patterns:
            if fnmatch(str(note.path), pattern):
                return False

        # If no includes specified, match all (after excludes)
        if not include_patterns:
            return True

        # Check includes
        for pattern in include_patterns:
            if fnmatch(str(note.path), pattern):
                return True

        return False


class TagFilter(Filter):
    """Filter based on tags"""

    def matches(self, note: Note) -> bool:
        has_any = self.params.get('has_any', [])
        has_all = self.params.get('has_all', [])
        has_none = self.params.get('has_none', [])

        note_tags = set(note.tags)

        # Check has_any
        if has_any:
            if not any(tag in note_tags for tag in has_any):
                return False

        # Check has_all
        if has_all:
            if not all(tag in note_tags for tag in has_all):
                return False

        # Check has_none
        if has_none:
            if any(tag in note_tags for tag in has_none):
                return False

        return True


class PropertyFilter(Filter):
    """Filter based on frontmatter properties"""

    def matches(self, note: Note) -> bool:
        key = self.params.get('key')
        value = self.params.get('value')
        exists = self.params.get('exists', None)

        if key is None:
            return True

        prop_value = note.properties.get(key)

        if exists is not None:
            return (prop_value is not None) == exists

        if value is not None:
            return prop_value == value

        return prop_value is not None


class ChangeTypeFilter(Filter):
    """Filter based on change type"""

    def matches(self, note: Note) -> bool:
        types = self.params.get('types', [])
        if not types:
            return True
        return note.change_type.value in types


class CompositeFilter(Filter):
    """Combine multiple filters with AND/OR logic"""

    def __init__(self, params: Dict[str, Any], filter_factory):
        super().__init__(params)
        self.operator = params.get('operator', 'and')
        self.filters = [
            filter_factory.create(f['type'], f['params'])
            for f in params.get('filters', [])
        ]

    def matches(self, note: Note) -> bool:
        if not self.filters:
            return True

        if self.operator == 'and':
            return all(f.matches(note) for f in self.filters)
        else:  # or
            return any(f.matches(note) for f in self.filters)


class FilterFactory:
    """Factory for creating filter instances"""

    def __init__(self):
        self.filter_classes = {
            'path': PathFilter,
            'tag': TagFilter,
            'property': PropertyFilter,
            'change_type': ChangeTypeFilter,
            'composite': CompositeFilter,
        }

    def create(self, filter_type: str, params: Dict[str, Any]) -> Filter:
        """Create a filter instance"""
        if filter_type == 'composite':
            return CompositeFilter(params, self)
        filter_class = self.filter_classes.get(filter_type)
        if not filter_class:
            raise ValueError(f"Unknown filter type: {filter_type}")
        return filter_class(params)


class FilterConfig:
    """Represents a filter configuration from a .jenny directory (Obsidian Actions config)"""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.script_dir = config_path.parent
        self.config = self._load_config()
        self.filter = self._build_filter()

    def _load_config(self) -> Dict[str, Any]:
        """Load filter.yaml"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _build_filter(self) -> Optional[Filter]:
        """Build filter from config"""
        filters = self.config.get('filters', [])
        if not filters:
            return None

        factory = FilterFactory()
        if len(filters) == 1:
            return factory.create(filters[0]['type'], filters[0]['params'])
        else:
            # Multiple filters - combine with AND
            return CompositeFilter({
                'operator': 'and',
                'filters': filters
            }, factory)

    def matches(self, note: Note) -> bool:
        """Check if note matches this filter"""
        if not self.config.get('enabled', True):
            return False
        if self.filter is None:
            return True
        return self.filter.matches(note)

    def get_script_path(self) -> Path:
        """Get the script path relative to the filter directory"""
        script = self.config.get('script', './script.sh')
        # If script is a command like "python3 script.py", extract the file part
        if ' ' in script and not script.startswith('./'):
            parts = script.split()
            if parts[0] in ['python', 'python3', 'python2']:
                script = parts[1]
        return self.script_dir / script
    
    def get_script_command(self) -> Optional[List[str]]:
        """Get the script command as a list (for subprocess execution).
        Returns None if script is just a file path, or a list like ['python3', 'script.py'] if it's a command."""
        script = self.config.get('script', './script.sh')
        # If script contains spaces and doesn't start with ./, it might be a command
        if ' ' in script and not script.startswith('./'):
            # Split into command parts
            parts = script.split()
            # If first part is python/python3, treat as command
            if parts[0] in ['python', 'python3', 'python2']:
                # Resolve the script file path
                script_file = self.script_dir / parts[1]
                return [parts[0], str(script_file)]
        return None


def load_filter_configs(jenny_dir: Path) -> List[FilterConfig]:
    """Load all filter configurations from .jenny directory"""
    configs = []
    if not jenny_dir.exists():
        logger.warning(f"Obsidian Actions config directory not found: {jenny_dir}")
        return configs

    for item in jenny_dir.iterdir():
        if item.is_dir():
            filter_yaml = item / 'filter.yaml'
            if filter_yaml.exists():
                try:
                    config = FilterConfig(filter_yaml)
                    configs.append(config)
                    logger.info(f"Loaded filter config: {item.name}")
                except Exception as e:
                    logger.error(f"Error loading filter config from {item}: {e}")

    return configs


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert date/datetime objects to strings for JSON serialization"""
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    else:
        return obj


def note_to_dict(note: Note) -> Dict[str, Any]:
    """Convert Note to dictionary for JSON serialization"""
    # Ensure tags is a list of strings
    tags_list = [str(tag) for tag in note.tags] if note.tags else []
    return {
        'title': note.title,
        'path': str(note.path),
        'content': note.content,
        'tags': tags_list,
        'properties': make_json_serializable(note.properties),
        'change_type': note.change_type.value,
        'commit_sha': note.commit_sha,
        'commit_message': note.commit_message
    }


def execute_script(script_path: Path, json_file: Path, script_command: Optional[List[str]] = None) -> bool:
    """Execute a script with the JSON file path as argument"""
    import subprocess
    
    # Determine how to execute the script
    if script_command:
        # Use provided command (e.g., ['python3', 'script.py'])
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        cmd = script_command + [str(json_file)]
        cwd = script_path.parent
        logger.info(f"Executing command: {' '.join(cmd)}")
    elif script_path.suffix == '.py':
        # Python script - run with python3
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        cmd = ['python3', str(script_path), str(json_file)]
        cwd = script_path.parent
        logger.info(f"Executing Python script: {' '.join(cmd)}")
    else:
        # Shell script or other executable
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        if not script_path.is_file():
            logger.error(f"Script path is not a file: {script_path}")
            return False

        # Make script executable
        script_path.chmod(0o755)
        cmd = [str(script_path), str(json_file)]
        cwd = script_path.parent
        logger.info(f"Executing script: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout:
            logger.info(f"Script output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Script failed with exit code {e.returncode}")
        if e.stderr:
            logger.error(f"Script error: {e.stderr}")
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        logger.error("Usage: obsidian_actions_runner.py <jenny_dir>")
        sys.exit(1)

    jenny_dir = Path(sys.argv[1]).resolve()
    repo_path = Path('.').resolve()

    logger.info(f"Obsidian Actions config directory: {jenny_dir}")
    logger.info(f"Repository path: {repo_path}")

    # Initialize components
    git_detector = GitDetector(repo_path)
    markdown_parser = MarkdownParser()

    # Load filter configurations
    filter_configs = load_filter_configs(jenny_dir)
    if not filter_configs:
        logger.warning("No filter configurations found")
        sys.exit(0)

    logger.info(f"Loaded {len(filter_configs)} filter configuration(s)")

    # Detect changed files
    # In GitHub Actions, we can use GITHUB_SHA and GITHUB_BEFORE
    before_sha = None
    after_sha = None

    import os
    github_before = os.environ.get('GITHUB_BEFORE')
    github_sha = os.environ.get('GITHUB_SHA')

    if github_before and github_sha:
        before_sha = github_before if github_before != '0000000000000000000000000000000000000000' else None
        after_sha = github_sha
        logger.info(f"Using GitHub context: {before_sha}..{after_sha}")
    else:
        logger.info("No GitHub context found, using HEAD and HEAD~1")

    changes = git_detector.get_changed_files(before_sha, after_sha)
    if not changes:
        logger.info("No markdown file changes detected")
        sys.exit(0)

    logger.info(f"Found {len(changes)} changed markdown file(s)")

    # Parse notes
    notes = []
    for change in changes:
        if change['change_type'] == ChangeType.DELETED:
            # For deleted files, we can't parse content
            # Create a minimal note
            note = Note(
                title=change['file_path'].stem,
                path=str(change['file_path']),
                content="",
                tags=[],
                properties={},
                change_type=change['change_type'],
                commit_sha=change['commit_sha'],
                commit_message=change['commit_message']
            )
            notes.append(note)
        else:
            try:
                content = git_detector.get_file_content(
                    change['file_path'],
                    change['commit_sha']
                )
                metadata = markdown_parser.parse(content, change['file_path'])

                note = Note(
                    title=metadata['title'],
                    path=str(change['file_path']),
                    content=metadata['content'],
                    tags=metadata['tags'],
                    properties=metadata['properties'],
                    change_type=change['change_type'],
                    commit_sha=change['commit_sha'],
                    commit_message=change['commit_message']
                )
                notes.append(note)
            except Exception as e:
                logger.error(f"Error parsing {change['file_path']}: {e}")
                continue

    # Evaluate filters and execute scripts
    total_matches = 0
    total_executions = 0

    for filter_config in filter_configs:
        logger.info(f"Evaluating filter: {filter_config.config.get('name', filter_config.config_path.parent.name)}")

        for note in notes:
            if filter_config.matches(note):
                total_matches += 1
                logger.info(f"Note matched: {note.path}")

                # Write note to JSON file
                note_dict = note_to_dict(note)
                logger.info(f"Note tags before serialization: {note.tags} (type: {type(note.tags)})")
                logger.info(f"Note dict tags: {note_dict.get('tags')} (type: {type(note_dict.get('tags'))})")
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(note_dict, f, indent=2)
                    json_file = Path(f.name)

                try:
                    # Execute script
                    script_path = filter_config.get_script_path()
                    script_command = filter_config.get_script_command()
                    logger.info(f"Executing script: {script_path}")
                    if execute_script(script_path, json_file, script_command):
                        total_executions += 1
                    else:
                        logger.error(f"Script execution failed: {script_path}")
                finally:
                    # Clean up temp file
                    json_file.unlink()

    logger.info("=" * 60)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total notes: {len(notes)}")
    logger.info(f"Total filter configs: {len(filter_configs)}")
    logger.info(f"Total matches: {total_matches}")
    logger.info(f"Total script executions: {total_executions}")

    if total_executions == 0 and total_matches > 0:
        logger.warning("Some matches occurred but no scripts were executed successfully")
        sys.exit(1)


if __name__ == '__main__':
    main()
