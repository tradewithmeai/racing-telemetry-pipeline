"""IO utilities for file operations and data persistence."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import orjson


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if needed.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def compute_file_hash(file_path: str | Path, algorithm: str = "sha256") -> str:
    """Compute hash of file contents.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('sha256', 'md5', etc.)

    Returns:
        Hex digest of file hash
    """
    hasher = hashlib.new(algorithm)

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def save_json(data: Dict[str, Any], file_path: str | Path, pretty: bool = True) -> None:
    """Save data as JSON file.

    Args:
        data: Data to save
        file_path: Output file path
        pretty: Whether to pretty-print (indent)
    """
    path_obj = Path(file_path)
    ensure_dir(path_obj.parent)

    if pretty:
        # Use orjson for speed but format with standard json for readability
        with open(path_obj, "w") as f:
            json.dump(data, f, indent=2, default=str)
    else:
        # Use orjson for fast serialization
        with open(path_obj, "wb") as f:
            f.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))


def load_json(file_path: str | Path) -> Dict[str, Any]:
    """Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data
    """
    with open(file_path, "rb") as f:
        return orjson.loads(f.read())


def save_parquet(
    df: Any,  # pandas or polars DataFrame
    file_path: str | Path,
    compression: str = "snappy",
    **kwargs,
) -> None:
    """Save DataFrame to Parquet.

    Args:
        df: DataFrame (pandas or polars)
        file_path: Output path
        compression: Compression codec
        **kwargs: Additional arguments for to_parquet
    """
    path_obj = Path(file_path)
    ensure_dir(path_obj.parent)

    # Detect DataFrame type
    if hasattr(df, "to_parquet"):
        # Pandas or Polars
        df.to_parquet(path_obj, compression=compression, **kwargs)
    else:
        raise TypeError(f"Unsupported DataFrame type: {type(df)}")


def get_file_size_mb(file_path: str | Path) -> float:
    """Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    size_bytes = Path(file_path).stat().st_size
    return size_bytes / (1024 * 1024)


def list_files_recursive(
    directory: str | Path, pattern: str = "*", max_depth: Optional[int] = None
) -> list[Path]:
    """List files recursively in directory.

    Args:
        directory: Root directory
        pattern: Glob pattern (e.g., "*.csv")
        max_depth: Maximum depth to search (None = unlimited)

    Returns:
        List of matching file paths
    """
    dir_path = Path(directory)

    if max_depth is None:
        return list(dir_path.rglob(pattern))
    else:
        # Manual depth control
        files = []

        def _search(path: Path, depth: int):
            if depth > max_depth:
                return

            for item in path.iterdir():
                if item.is_file() and item.match(pattern):
                    files.append(item)
                elif item.is_dir():
                    _search(item, depth + 1)

        _search(dir_path, 0)
        return files
