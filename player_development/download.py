"""Download StatsBomb open data (JSON) into a local cache.

The data itself is never committed to this repository: the StatsBomb Public Data User Agreement
does not allow redistribution, so every user downloads it from StatsBomb's GitHub repository.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import config


def fetch(url: str, dest: Path, retries: int = 6) -> Path:
    """Download ``url`` to ``dest`` unless a non-empty copy is already cached."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
            return dest
        except urllib.error.HTTPError as err:
            if err.code == 404:
                raise
            time.sleep(2 ** attempt)  # GitHub answers 429 when requests come too fast
        except urllib.error.URLError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {url} after {retries} attempts")


def match_list_path(season_id: int, raw_dir: Path | None = None) -> Path:
    raw_dir = raw_dir or config.RAW_DIR
    return raw_dir / "matches" / str(config.COMPETITION_ID) / f"{season_id}.json"


def download(workers: int = 4, raw_dir: Path | None = None) -> list[int]:
    """Fetch match lists, events and lineups for every configured season; returns match ids."""
    raw_dir = raw_dir or config.RAW_DIR
    fetch(f"{config.BASE_URL}/competitions.json", raw_dir / "competitions.json")
    match_ids: list[int] = []
    for season_id in config.SEASONS:
        url = f"{config.BASE_URL}/matches/{config.COMPETITION_ID}/{season_id}.json"
        path = fetch(url, match_list_path(season_id, raw_dir))
        match_ids += [m["match_id"] for m in json.loads(path.read_text(encoding="utf-8"))]
    jobs = [
        (f"{config.BASE_URL}/{kind}/{match_id}.json", raw_dir / kind / f"{match_id}.json")
        for match_id in match_ids
        for kind in ("events", "lineups")
    ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda job: fetch(*job), jobs))
    return match_ids
