from pathlib import Path

def ensure_dir(path: str):
    """
    Create a directory if it does not already exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)

