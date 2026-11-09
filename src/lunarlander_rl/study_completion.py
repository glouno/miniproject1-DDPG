"""Validate saved completion without importing the training dependencies."""
import json
from pathlib import Path


def summary_error(directory: Path, expected_transitions: int) -> str | None:
    """Return an actionable reason when a saved summary cannot prove completion."""
    try:
        summary = json.loads((directory / 'summary.json').read_text())
    except FileNotFoundError:
        return 'summary.json is missing'
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 'summary.json is unreadable or malformed; inspect the retained artifact'
    if not isinstance(summary, dict) or summary.get('status') != 'complete':
        return 'summary.json does not report status=complete'
    transitions = summary.get('transitions')
    if type(transitions) is not int or transitions != expected_transitions:
        return f'summary.json must report exactly {expected_transitions} transitions'
    return None
