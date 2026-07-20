
from pathlib import Path

def get_name(path: str) -> str:
    """Get the name of a file or directory from a path."""
    return Path(path).name

def print_header(string):
    print(f"\n\n{'='*50}\n{string}\n{'='*50}")
