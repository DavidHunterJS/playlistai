"""
Subsonic Playlist Exporter

Exports all playlists from a Subsonic server to M3U files so that
phase1_playlists.py can import them into the PlaylistAI database.

Run on the seedbox before phase1_playlists.py:
    python subsonic_export.py --out-dir ~/playlistai/playlists

Environment variables (or set in .env):
    SUBSONIC_URL   — e.g. http://localhost:12851/subsonic
    SUBSONIC_USER
    SUBSONIC_PASS
"""

import os
import re
import logging
import argparse
import urllib.request
import urllib.parse
import json
from pathlib import Path

from config import SUBSONIC_URL, SUBSONIC_USER, SUBSONIC_PASS, SUBSONIC_API_VERSION, SUBSONIC_CLIENT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("subsonic_export.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def _api(endpoint: str, **params) -> dict:
    """Make a Subsonic REST API call and return the inner response object."""
    base = SUBSONIC_URL.rstrip("/")
    qs = urllib.parse.urlencode({
        "u": SUBSONIC_USER,
        "p": SUBSONIC_PASS,
        "v": SUBSONIC_API_VERSION,
        "c": SUBSONIC_CLIENT,
        "f": "json",
        **params,
    })
    url = f"{base}/rest/{endpoint}?{qs}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.loads(resp.read())
    outer = data["subsonic-response"]
    if outer["status"] != "ok":
        raise RuntimeError(f"Subsonic error: {outer.get('error')}")
    return outer


def _safe_filename(name: str) -> str:
    """Strip non-filesystem-safe characters from a playlist name."""
    name = re.sub(r"[^\w\s\-.]", "", name).strip()
    name = re.sub(r"\s+", " ", name)
    return name or "playlist"


def export_playlists(out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    log.info("Fetching playlist list from %s …", SUBSONIC_URL)
    data = _api("getPlaylists")
    playlists = data.get("playlists", {}).get("playlist", [])

    if not playlists:
        log.warning("No playlists found.")
        return

    log.info("Found %d playlists", len(playlists))

    exported = 0
    skipped = 0

    for pl in playlists:
        pl_id = pl["id"]
        pl_name = pl.get("name", f"playlist_{pl_id}")
        song_count = pl.get("songCount", 0)

        if song_count == 0:
            log.info("Skipping empty playlist: %s", pl_name)
            skipped += 1
            continue

        log.info("Exporting '%s' (%d songs) …", pl_name, song_count)

        try:
            detail = _api("getPlaylist", id=pl_id)
            entries = detail.get("playlist", {}).get("entry", [])
        except Exception as e:
            log.warning("Failed to fetch playlist %s (%s): %s", pl_id, pl_name, e)
            skipped += 1
            continue

        if not entries:
            log.info("Playlist '%s' returned no entries — skipping", pl_name)
            skipped += 1
            continue

        filename = _safe_filename(pl_name) + ".m3u"
        out_path = Path(out_dir) / filename

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("#EXTM3U\n")
            for entry in entries:
                path = entry.get("path", "")
                title = entry.get("title", "")
                artist = entry.get("artist", "")
                duration = entry.get("duration", -1)
                display = f"{artist} - {title}" if artist and title else (title or path)
                f.write(f"#EXTINF:{duration},{display}\n")
                f.write(f"{path}\n")

        log.info("  → %s (%d entries)", out_path, len(entries))
        exported += 1

    log.info("Export complete — %d playlists exported, %d skipped", exported, skipped)
    print(f"\nPlaylists written to: {out_dir}")
    print(f"  Exported : {exported}")
    print(f"  Skipped  : {skipped} (empty)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Subsonic playlists to M3U files")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.expanduser("~"), "playlistai", "playlists"),
        help="Directory to write M3U files into (default: ~/playlistai/playlists)",
    )
    args = parser.parse_args()
    export_playlists(args.out_dir)
