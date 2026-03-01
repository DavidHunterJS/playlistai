# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PlaylistAI analyzes a 249,000-song music collection on a Bytesized seedbox to surface songs the owner will love but has never heard. It trains a binary "David likes this" classifier using ~8,000 playlisted songs as positives and ~117,000 explored-but-never-playlisted songs as negatives, then runs inference on ~124,000 unexplored songs and pushes ranked discovery playlists to Subsonic/DSub.

**Infrastructure:**
- **Seedbox** (Bytesized): hosts the collection, runs Subsonic, SSH access. Phases 1–2 run here.
- **RunPod GPU** (RTX 3090 @ $0.50/hr): temporary rental for audio processing and training. Phases 3–5 run here. Music files accessed via rclone SFTP mount.
- **SQLite**: single portable database file that moves between environments; the project's source of truth.

**Collection breakdown:**

| Category | Count | Role |
|---|---|---|
| Total collection | 249,000 | — |
| Playlisted (positive labels) | ~8,000 | Training: LIKE |
| Explored, not playlisted (negative pool) | ~117,000 | Training: DISLIKE |
| Unexplored (inference targets) | ~124,000 | Prediction |

**Genre playlists:** House (~583), Rap/Hip-Hop (~1,479+), Rock/Metal, Pop, EDM/Electronic.

## Environment Setup

```bash
# Install Phase 1 dependencies (seedbox — no GPU)
pip install -r requirements.txt

# GPU phases (RunPod only — install separately)
pip install tensorflow tensorflow-hub librosa pandas numpy tqdm mutagen

# Configure environment
cp .env.example .env
export $(grep -v '^#' .env | xargs)
```

All configuration lives in `config.py`, overridable via environment variables: `PLAYLISTAI_MUSIC_ROOT`, `PLAYLISTAI_DB_PATH`, `PLAYLISTAI_PLAYLIST_DIR`, `PLAYLISTAI_EXPLORED_CUTOFF`, `SUBSONIC_URL`, `SUBSONIC_USER`, `SUBSONIC_PASS`.

## Phase 1 Pipeline (Seedbox — Implemented)

Run these scripts in order:

```bash
# Export Subsonic playlists to M3U files first (playlists live in Subsonic's internal DB)
python subsonic_export.py --out-dir ~/playlistai/playlists

# Initialize DB and scan music files (ID3 tags, folder-date parsing, genre extraction)
python phase1_scan.py [--music-root ~/media/Audio] [--cutoff 2021-08-31]

# Import the exported M3U playlists into the DB
python phase1_playlists.py [--playlist-dir ~/playlistai/playlists] [--music-root ~/media/Audio]

# Print metadata coverage report
python phase1_report.py

# Initialize DB schema alone (without scanning)
python db.py
```

**Seedbox paths:**
- Music root: `~/media/Audio` (contains `mp3/` and `soundcheck/` subdirectories)
- `mp3/`: ~58,700 songs — all `is_explored=1` (via `EXPLORED_FOLDER_NAMES`)
- `soundcheck/`: ~178,800 songs — `is_explored` determined by folder date vs `EXPLORED_CUTOFF`
- Subsonic API: `http://localhost:12851/subsonic` (user: `admin`)
- Subsonic playlist folder: Subsonic stores playlists internally; use `subsonic_export.py` to dump them

**Key files:**

| File | Purpose |
|---|---|
| `config.py` | All constants and tunable parameters (genre maps, ML thresholds, paths) |
| `db.py` | SQLite schema, `init_db()`, `get_db()` context manager, `get_connection()` |
| `phase1_scan.py` | Steps 1.2–1.4: walk filesystem, extract ID3/mutagen tags, folder-name genre cascade, filename metadata fallback |
| `phase1_playlists.py` | Step 1.5: parse M3U/M3U8 files, link songs → `playlists` + `playlist_songs` tables |
| `phase1_report.py` | Step 1.6: collection inventory, genre coverage, metadata quality stats |
| `phase2_prep.py` | Assemble positives + stratified negatives, write 80/20 train/val split to `training_manifest` |
| `subsonic_export.py` | Export all Subsonic playlists to M3U files (run before `phase1_playlists.py`) |
| `phase3_extract.py` | RunPod: extract YAMNet embeddings + genre classification for training_manifest songs |
| `phase4_train.py` | RunPod: train binary classifier, save model + norm stats, write genre centroids to DB |
| `phase5_inference.py` | RunPod: extract embeddings for 124K unexplored songs, genre fallback, inference, write M3U playlists |
| `phase6_push.py` | Seedbox: resolve M3U paths to Subsonic IDs, create/update discovery playlists via API |
| `phase6_sync.py` | Seedbox: pull starred songs from Subsonic → feedback table; prints retrain guidance when ≥50 stars |

## Full Phase Plan (Phases 5–6 Not Yet Implemented)

```
Phase 0  Prerequisites: RunPod account, SSH verified, genre keyword dict sampled from folder names
Phase 1  Seedbox: File scan → SQLite population → M3U playlist import (DONE)
Phase 2  Training prep: assemble 8K positives + 8K stratified negatives, 80/20 train/val split (DONE)
Phase 3  RunPod: Dual extraction — 1024-dim YAMNet embeddings + genre classification for 16K training songs (~10–14 hrs)
Phase 4  RunPod: Train dense classifier (1024→256→128→1), compute genre centroids (DONE)
Phase 5  RunPod: Extract embeddings for 124K unexplored songs (~70–100 hrs), apply embedding-similarity genre fallback, run predictions, generate playlists (DONE)
Phase 6  Seedbox: Push playlists to Subsonic via API, collect star-based feedback, periodic retrain (DONE)
```

### Running Phase 6 on the seedbox

```bash
# 6A — push discovery playlists (run once after copying playlists from RunPod)
scp -r root@RUNPOD_IP:/workspace/playlistai/playlists ~/playlistai/playlists
python phase6_push.py --playlist-dir ~/playlistai/playlists

# 6B — sync Subsonic stars to feedback table (run periodically)
python phase6_sync.py

# Full retrain cycle (when phase6_sync.py reports ≥50 new stars):
scp ~/playlistai/playlistai.db root@RUNPOD_IP:/workspace/playlistai/
# On RunPod:
python phase2_prep.py && python phase3_extract.py && python phase4_train.py
python phase5_inference.py --skip-extraction
# Back on seedbox:
scp -r root@RUNPOD_IP:/workspace/playlistai/playlists ~/playlistai/playlists
python phase6_push.py
```

### Running Phase 5 on RunPod

```bash
# Full run (step 1 takes ~70–100 hrs; fully resumable — restart freely)
python phase5_inference.py --db-path playlistai.db

# Skip embedding extraction (if all embeddings already in DB)
python phase5_inference.py --db-path playlistai.db --skip-extraction

# Regenerate playlists only (from existing predictions)
python phase5_inference.py --db-path playlistai.db --skip-extraction --skip-inference
```

**Playlists generated** (written to `playlists/` as M3U files with seedbox absolute paths):
- `Discover Master.m3u` — top 200 across all genres
- `Discover House/Rap/Rock-Metal/Pop/EDM-Electronic.m3u` — top 50 per genre
- `Discover Mystery.m3u` — top 50 ungenred songs
- `Discover Borderline.m3u` — top 50 scored 0.45–0.55 (uncertain but interesting)

### Running Phase 4 on RunPod

```bash
# After phase3_extract.py completes and DB is in /workspace/playlistai/
python phase4_train.py [--db-path playlistai.db] [--model-dir ./model] [--epochs 200] [--batch-size 256]
```

Outputs written to `model/`:
- `playlistai_model.keras` — trained Keras model
- `norm_stats.npz` — training-set mean + std (required by Phase 5 inference)
- `model_version.txt` — version string used as key in `predictions` table
- `checkpoint.keras` — best checkpoint (same weights as final model after early stopping)

**Phase 2 — Negative sampling:** stratified across genres (matching positive distribution), `NEGATIVE_SAMPLE_SIZE = 8000`, `RANDOM_SEED = 42`, results written to `training_manifest` table with `label` (0/1) and `split` ('train'/'val') columns.

**Phase 3 — Embedding extraction:** load audio at 16kHz mono via librosa, take middle `CLIP_SECONDS = 60` seconds, run YAMNet, mean-pool frame embeddings to 1024-dim vector. Checkpoint/resume every `CHECKPOINT_EVERY = 500` songs (critical — RunPod can disconnect).

**Phase 4 — Model architecture:**
```
Input:   1024-dim embedding
Dense:   256 units, ReLU
Dropout: 0.5
Dense:   128 units, ReLU
Dropout: 0.5
Output:  1 unit, sigmoid
```
Adam optimizer, binary cross-entropy, early stopping patience=15, target >75% validation accuracy. Normalize embeddings using training-set statistics only; save mean/std vectors for reuse at inference time. After training, compute mean embedding per genre from playlisted songs → `genre_centroids` table.

**Phase 5 — Discovery playlists generated:**
- Master Discovery (top 200, all genres)
- Discover: House / Rap / Rock/Metal / Pop / EDM/Electronic (top 50 each)
- Discover: Mystery (top 50 ungenred songs)
- Borderline (top 50 scoring 0.45–0.55)

**Phase 6 — Feedback loop:** star in DSub → Subsonic API → `feedback` table → retrain (fast, embeddings already in DB). Each cycle expected to improve accuracy 2–5%.

## Architecture

### Database Schema

SQLite with WAL mode + foreign keys. Core tables:
- **`songs`** — one row per audio file; central table for all phases
- **`playlists`** / **`playlist_songs`** — M3U playlists and their song membership (positive training labels)
- **`training_manifest`** — 16K rows with `label`, `split`; the training/val dataset
- **`predictions`** — per-song ML scores keyed by `model_version`
- **`feedback`** — binary user ratings (0/1) from Subsonic star sync
- **`genre_centroids`** — mean embedding per genre, used for Layer 4 genre fallback

### Genre Resolution Cascade

Genre is **only a presentation layer** for organizing playlists — the ML model operates purely on embeddings and ignores genre. The cascade fills `genre_tag` + `genre_source`:

| Layer | `genre_source` | When | Expected coverage |
|---|---|---|---|
| 1. ID3 tag | `'id3'` | Phase 1 scan | 40–60% |
| 2. Folder name keywords | `'folder'` | Phase 1 scan | +15–20% |
| 3. YAMNet classification | `'yamnet'` | Phase 3 & 5 | +15–25% |
| 4. Embedding similarity to centroids | `'similarity'` | Phase 5 post-training | +3–7% |
| — | `'unknown'` | Remaining | 3–10% |

Layer 2 uses `FOLDER_GENRE_MAP` in `config.py`; longer keywords take priority (sorted by length descending). Layer 3 uses `YAMNET_GENRE_MAP` and only assigns genre if confidence ≥ `YAMNET_CONFIDENCE_THRESHOLD` (0.3). Layer 4 assigns genre if cosine similarity > `SIMILARITY_MIN` (0.7) AND exceeds second-best by > `SIMILARITY_MARGIN` (0.1). Songs that remain ungenred still get predicted and appear in Master/Mystery playlists.

### Explored vs. Unexplored

Folder dates are parsed from path segments (supports `YYYY-WW`, `YYYY-MM-DD`, `YYYY-MM`, `YYYY`). Songs with `folder_date < EXPLORED_CUTOFF` get `is_explored = 1`. The explored-but-not-playlisted pool (~109K songs) is the negative sampling pool for Phase 2.

### Subsonic Integration

Subsonic API v`1.16.1`, client name `"playlistai"`. Credentials via `SUBSONIC_URL` / `SUBSONIC_USER` / `SUBSONIC_PASS` env vars (configured in `config.py`). Phase 6 uses `createPlaylist` and `getStarred` endpoints. File paths must be resolved to Subsonic song IDs before playlist creation.
