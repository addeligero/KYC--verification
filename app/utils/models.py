import os
from typing import Iterable, List, Set, Union

import requests


def _looks_like_git_lfs_pointer(data: bytes) -> bool:
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return False
    return text.startswith("version https://git-lfs.github.com/spec/v1") and "oid " in text


def _derive_media_url(url: str) -> str | None:
    """Convert a raw.githubusercontent.com URL to media.githubusercontent.com variant."""
    prefix = "https://raw.githubusercontent.com/"
    if url.startswith(prefix):
        return "https://media.githubusercontent.com/media/" + url[len(prefix):]
    # GitHub web URLs sometimes use /raw/ path
    alt_prefix = "https://github.com/"
    if url.startswith(alt_prefix) and "/raw/" in url:
        head, tail = url[len(alt_prefix):].split("/raw/", 1)
        return f"https://media.githubusercontent.com/media/{head}/{tail}"
    return None


def ensure_model(file_name: str, urls: Union[str, Iterable[str]], directory: str = "models") -> str:
    """
    Ensure a model file exists locally.
    - If already present and non-empty: return its path.
    - Otherwise, try to download from one or more candidate URLs in order.
    - Raises a clear error with manual download instructions if all sources fail.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, file_name)

    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path

    initial_candidates: List[str] = [urls] if isinstance(urls, str) else list(urls)
    candidates: List[str] = list(initial_candidates)
    attempted: Set[str] = set()

    errors = []
    while candidates:
        url = candidates.pop(0)
        if url in attempted:
            continue
        attempted.add(url)
        try:
            with requests.get(url, stream=True, timeout=90) as r:
                r.raise_for_status()
                data = r.content
            if not data:
                errors.append(f"Downloaded empty response from {url}")
                continue

            if len(data) < 1024 and _looks_like_git_lfs_pointer(data):
                media_url = _derive_media_url(url)
                if media_url and media_url not in attempted and media_url not in candidates:
                    candidates.insert(0, media_url)
                    errors.append(
                        "Encountered Git LFS pointer; retrying via media URL: "
                        + media_url
                    )
                    continue

            with open(path, "wb") as f:
                f.write(data)

            if os.path.getsize(path) < 1024:  # too small to be a real ONNX file
                errors.append(f"Downloaded file too small from {url}")
                try:
                    os.remove(path)
                except OSError:
                    pass
                continue
            return path
        except Exception as e:
            errors.append(f"{url} -> {e}")

    msg = (
        f"Could not download model '{file_name}'. Tried:\n  - "
        + "\n  - ".join(errors)
        + "\n\nManual fix:\n"
          f"1) Create the models folder if missing: mkdir -p {directory}\n"
          f"2) Download the file with your browser or curl and place it at: {path}\n"
          f"3) Or set an env var to your own mirror URL.\n"
    )
    raise RuntimeError(msg)