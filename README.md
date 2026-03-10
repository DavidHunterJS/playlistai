# PlaylistAI

A personal music discovery system that trains a binary "David likes this" classifier on a 249,000-song collection, then surfaces songs you'll love but haven't heard yet.

## How It Works

PlaylistAI analyzes your Subsonic music library to learn your taste from playlist history, then ranks unexplored songs by predicted likelihood of enjoyment and pushes discovery playlists back to Subsonic/DSub.

**Training signal:**
- ~8,000 playlisted songs → **LIKE** (positives)
- ~117,000 explored-but-never-playlisted songs → **DISLIKE** (negatives)
- ~124,000 unexplored songs → **inference targets**

**Model:** 1024-dim YAMNet audio embeddings → dense classifier (1024→256→128→1) → per-song like probability.

**Feedback loop:** Star songs in DSub → sync to DB → retrain → improved recommendations.

## Infrastructure

| Environment | Role |
|---|---|
| **Seedbox** (Bytesized) | Hosts the music collection and Subsonic. Runs Phases 1, 2, 6. |
| **RunPod GPU** (RTX 3090) | Temporary rental for audio processing and training. Runs Phases 3–5. |
| **SQLite** | Single portable DB file that moves between environments. |

## Pipeline

```
Phase 1  Seedbox: Scan filesystem → SQLite, import M3U playlists
Phase 2  Seedbox: Assemble training set (positives + stratified negatives, 80/20 split)
Phase 3  RunPod: Extract YAMNet embeddings for 16K training songs (~10–14 hrs)
Phase 4  RunPod: Train classifier, compute genre centroids
Phase 5  RunPod: Extract embeddings for 124K unexplored songs (~70–100 hrs), run inference, generate playlists
Phase 6  Seedbox: Push playlists to Subsonic, collect star feedback
```

## Setup

```bash
# Clone and configure
git clone <repo>
cd playlistai
cp .env.example .env
# Edit .env with your Subsonic credentials and paths

# Phase 1 dependencies (seedbox — no GPU)
pip install -r requirements.txt

# Phase 3–5 dependencies (RunPod only)
pip install tensorflow[and-cuda] tensorflow-hub librosa pandas numpy tqdm mutagen
```

All config lives in `config.py`, overridable via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PLAYLISTAI_MUSIC_ROOT` | `~/media/Audio` | Root of music collection |
| `PLAYLISTAI_DB_PATH` | `playlistai.db` | SQLite database path |
| `PLAYLISTAI_PLAYLIST_DIR` | `~/playlistai/playlists` | M3U playlist directory |
| `PLAYLISTAI_EXPLORED_CUTOFF` | `2021-08-31` | Folders before this date = explored |
| `SUBSONIC_URL` | — | Subsonic server URL |
| `SUBSONIC_USER` | — | Subsonic username |
| `SUBSONIC_PASS` | — | Subsonic password |

## Running

### Phase 1 — Scan & Import (Seedbox)

```bash
# Export Subsonic playlists to M3U files
python subsonic_export.py --out-dir ~/playlistai/playlists

# Scan music files into DB (ID3 tags, folder-date parsing, genre extraction)
python phase1_scan.py [--music-root ~/media/Audio] [--cutoff 2021-08-31]

# Import M3U playlists (positive training labels)
python phase1_playlists.py [--playlist-dir ~/playlistai/playlists]

# Metadata coverage report
python phase1_report.py
```

### Phase 2 — Training Prep (Seedbox)

```bash
python phase2_prep.py
```

Assembles 8K positives + 8K stratified negatives with 80/20 train/val split into the `training_manifest` table. Starred feedback songs are automatically included as additional positives.

### Phase 3 — Embedding Extraction (RunPod)

```bash
python phase3_extract.py
```

Loads each training song at 16kHz mono, takes the middle 60 seconds, runs YAMNet, mean-pools frame embeddings to a 1024-dim vector. Checkpoints every 500 songs — safe to interrupt and resume.

### Phase 4 — Train Classifier (RunPod)

```bash
python phase4_train.py [--epochs 200] [--batch-size 256]
```

Outputs to `model/`:
- `playlistai_model.keras` — trained model
- `norm_stats.npz` — embedding mean/std for inference normalization
- `checkpoint.keras` — best validation checkpoint

Target: >75% validation accuracy.

### Phase 5 — Inference & Playlists (RunPod)

```bash
# Full run (embedding extraction takes ~70–100 hrs; fully resumable)
python phase5_inference.py --db-path playlistai.db

# Skip extraction if embeddings already in DB
python phase5_inference.py --db-path playlistai.db --skip-extraction

# Regenerate playlists only (from existing predictions)
python phase5_inference.py --db-path playlistai.db --skip-extraction --skip-inference
```

Playlists generated (M3U files with seedbox absolute paths):
- `Discover Master.m3u` — top 200 across all genres
- `Discover House/Rap/Rock-Metal/Pop/EDM-Electronic.m3u` — top 50 per genre
- `Discover Mystery.m3u` — top 50 ungenred songs
- `Discover Borderline.m3u` — top 50 scoring 0.45–0.55

### Phase 6 — Push to Subsonic & Feedback (Seedbox)

```bash
# Copy playlists from RunPod
scp -r root@RUNPOD_IP:/workspace/playlistai/playlists ~/playlistai/playlists

# Push discovery playlists to Subsonic
python phase6_push.py --playlist-dir ~/playlistai/playlists

# Sync Subsonic stars to feedback table (run periodically)
python phase6_sync.py
```

When `phase6_sync.py` reports ≥50 new stars, run a retrain cycle.

## Genre Resolution

Genre is a presentation layer for organizing playlists — the model operates purely on audio embeddings and ignores genre labels.

| Layer | Source | When | Coverage |
|---|---|---|---|
| 1. ID3 tag | `id3` | Phase 1 | 40–60% |
| 2. Folder name keywords | `folder` | Phase 1 | +15–20% |
| 3. YAMNet classification | `yamnet` | Phase 3 & 5 | +15–25% |
| 4. Embedding similarity to centroids | `similarity` | Phase 5 | +3–7% |
| — | `unknown` | Remaining | 3–10% |

## Database Schema

SQLite with WAL mode. Key tables:

| Table | Purpose |
|---|---|
| `songs` | One row per audio file; central table |
| `playlists` / `playlist_songs` | M3U playlists and song membership |
| `training_manifest` | 16K rows with `label` and `split` for training |
| `predictions` | Per-song ML scores keyed by model version |
| `feedback` | Star ratings from Subsonic sync |
| `genre_centroids` | Mean embedding per genre (used for genre fallback) |

## File Reference

| File | Purpose |
|---|---|
| `config.py` | All constants and tunable parameters |
| `db.py` | SQLite schema and connection helpers |
| `subsonic_export.py` | Export Subsonic playlists to M3U |
| `phase1_scan.py` | Filesystem scan, ID3 extraction, genre cascade |
| `phase1_playlists.py` | M3U import → playlist tables |
| `phase1_report.py` | Metadata coverage report |
| `phase2_prep.py` | Training set assembly with stratified sampling |
| `phase3_extract.py` | YAMNet embedding extraction |
| `phase4_train.py` | Classifier training |
| `phase5_inference.py` | Inference on unexplored songs, playlist generation |
| `phase6_push.py` | Push M3U playlists to Subsonic via API |
| `phase6_sync.py` | Sync Subsonic stars → feedback table |
