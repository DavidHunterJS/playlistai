"""
PlaylistAI configuration.
Edit these values before running on the seedbox.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
# Root of the music collection on the seedbox (or rclone mount point on RunPod)
MUSIC_ROOT = os.environ.get("PLAYLISTAI_MUSIC_ROOT", "/music")

# Where to store the SQLite database
DB_PATH = os.environ.get("PLAYLISTAI_DB_PATH", "playlistai.db")

# Directory containing M3U playlist files
PLAYLIST_DIR = os.environ.get("PLAYLISTAI_PLAYLIST_DIR", "/music/playlists")

# ── Collection parameters ──────────────────────────────────────────────────────
# Folder dates BEFORE this cutoff are considered "explored".
# Format: "YYYY-WW" (ISO week) or "YYYY-MM-DD" — set during Phase 0 by reviewing
# which dated folders you've actually listened through.
EXPLORED_CUTOFF = os.environ.get("PLAYLISTAI_EXPLORED_CUTOFF", "2021-08-31")

# Top-level folder names (under MUSIC_ROOT) that are unconditionally treated as
# explored, regardless of folder date.  The "mp3" folder contains all liked/heard
# music; "soundcheck" is the unexplored discovery target and is NOT listed here.
EXPLORED_FOLDER_NAMES: set[str] = {"mp3"}

# ── Audio formats to scan ──────────────────────────────────────────────────────
AUDIO_EXTENSIONS = {".mp3", ".flac", ".m4a", ".ogg", ".wav", ".aac", ".opus"}

# ── Genre keyword map for folder-name parsing (Layer 2 cascade) ───────────────
FOLDER_GENRE_MAP = {
    # House
    "deep house": "House",
    "tech house": "House",
    "progressive house": "House",
    "acid house": "House",
    "future house": "House",
    "tropical house": "House",
    "afro house": "House",
    "house music": "House",
    "house": "House",
    # Rap / Hip-Hop
    "hip hop": "Rap",
    "hip-hop": "Rap",
    "hiphop": "Rap",
    "trap music": "Rap",
    "trap": "Rap",
    "grime": "Rap",
    "rap": "Rap",
    # Rock / Metal
    "hard rock": "Rock/Metal",
    "heavy metal": "Rock/Metal",
    "thrash metal": "Rock/Metal",
    "death metal": "Rock/Metal",
    "black metal": "Rock/Metal",
    "nu metal": "Rock/Metal",
    "punk rock": "Rock/Metal",
    "post punk": "Rock/Metal",
    "grunge": "Rock/Metal",
    "metal": "Rock/Metal",
    "punk": "Rock/Metal",
    "rock": "Rock/Metal",
    # Pop
    "synth pop": "Pop",
    "synthpop": "Pop",
    "dance pop": "Pop",
    "indie pop": "Pop",
    "electropop": "Pop",
    "pop music": "Pop",
    "pop": "Pop",
    # EDM / Electronic
    "drum and bass": "EDM/Electronic",
    "d&b": "EDM/Electronic",
    "dnb": "EDM/Electronic",
    "dubstep": "EDM/Electronic",
    "breakbeat": "EDM/Electronic",
    "jungle": "EDM/Electronic",
    "downtempo": "EDM/Electronic",
    "ambient music": "EDM/Electronic",
    "ambient": "EDM/Electronic",
    "idm": "EDM/Electronic",
    "trance": "EDM/Electronic",
    "techno": "EDM/Electronic",
    "electronic music": "EDM/Electronic",
    "electronic": "EDM/Electronic",
    "edm": "EDM/Electronic",
}

# ── YAMNet genre mapping (Layer 3 cascade) ────────────────────────────────────
YAMNET_GENRE_MAP = {
    "Hip hop music": "Rap",
    "House music": "House",
    "Techno": "EDM/Electronic",
    "Electronic music": "EDM/Electronic",
    "Drum and bass": "EDM/Electronic",
    "Dubstep": "EDM/Electronic",
    "Trance music": "EDM/Electronic",
    "Ambient music": "EDM/Electronic",
    "Rock music": "Rock/Metal",
    "Heavy metal": "Rock/Metal",
    "Punk rock": "Rock/Metal",
    "Pop music": "Pop",
    "Rhythm and blues": "Pop",
    "Reggae": "Other",
    "Jazz": "Other",
    "Country": "Other",
    "Classical music": "Other",
}

YAMNET_CONFIDENCE_THRESHOLD = 0.3

# ── Embedding similarity thresholds (Layer 4 cascade) ────────────────────────
SIMILARITY_MIN = 0.7        # Minimum cosine similarity to assign a genre
SIMILARITY_MARGIN = 0.1     # Best similarity must exceed second-best by this much

# ── Training / ML ─────────────────────────────────────────────────────────────
NEGATIVE_SAMPLE_SIZE = 8000
RANDOM_SEED = 42
TRAIN_VAL_SPLIT = 0.8       # 80% train, 20% val
CLIP_SECONDS = 60           # Middle N seconds of each track for YAMNet
CHECKPOINT_EVERY = 500      # Save progress every N songs during extraction

# ── Subsonic API ──────────────────────────────────────────────────────────────
SUBSONIC_URL = os.environ.get("SUBSONIC_URL", "")
SUBSONIC_USER = os.environ.get("SUBSONIC_USER", "")
SUBSONIC_PASS = os.environ.get("SUBSONIC_PASS", "")
SUBSONIC_API_VERSION = "1.16.1"
SUBSONIC_CLIENT = "playlistai"
