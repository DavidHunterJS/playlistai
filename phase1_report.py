"""
Phase 1, Step 1.6: Metadata coverage report and validation.

Prints a summary of collection inventory and metadata quality.
Run after phase1_scan.py and phase1_playlists.py.

    python phase1_report.py
"""

from db import get_connection
from config import DB_PATH


def _fmt(n: int, total: int) -> str:
    pct = 100 * n / total if total else 0
    return f"{n:>8,}  ({pct:5.1f}%)"


def run_report():
    conn = get_connection(DB_PATH)

    total = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    if total == 0:
        print("Database is empty — run phase1_scan.py first.")
        return

    explored = conn.execute("SELECT COUNT(*) FROM songs WHERE is_explored = 1").fetchone()[0]
    unexplored = total - explored

    playlisted = conn.execute(
        "SELECT COUNT(DISTINCT song_id) FROM playlist_songs"
    ).fetchone()[0]

    multi = conn.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT song_id FROM playlist_songs GROUP BY song_id HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    print("=" * 60)
    print("PLAYLISTAI — PHASE 1 COVERAGE REPORT")
    print("=" * 60)

    print("\n── Collection Inventory ─────────────────────────────────")
    print(f"  Total songs in DB :  {total:>10,}")
    print(f"  Explored          :  {_fmt(explored, total)}")
    print(f"  Unexplored        :  {_fmt(unexplored, total)}")
    print(f"  Playlisted (pos)  :  {_fmt(playlisted, total)}")
    print(f"  In multiple lists :  {_fmt(multi, total)}")

    # Explored-but-not-playlisted (negative pool)
    neg_pool = conn.execute(
        """
        SELECT COUNT(*) FROM songs
        WHERE is_explored = 1
          AND song_id NOT IN (SELECT song_id FROM playlist_songs)
        """
    ).fetchone()[0]
    print(f"  Negative pool     :  {_fmt(neg_pool, total)}  (explored, not playlisted)")

    print("\n── Playlists ─────────────────────────────────────────────")
    playlists = conn.execute(
        "SELECT name, song_count FROM playlists ORDER BY song_count DESC"
    ).fetchall()
    if playlists:
        for pl in playlists:
            print(f"  {pl['name']:<30} {pl['song_count']:>6,} songs")
    else:
        print("  (no playlists imported yet — run phase1_playlists.py)")

    print("\n── Genre Coverage ────────────────────────────────────────")
    genre_sources = conn.execute(
        """
        SELECT
            COALESCE(genre_source, 'none') AS source,
            COUNT(*) AS cnt
        FROM songs
        GROUP BY source
        ORDER BY cnt DESC
        """
    ).fetchall()
    for row in genre_sources:
        label = {
            "id3": "ID3 tag",
            "folder": "Folder name",
            "yamnet": "YAMNet (Phase 3)",
            "similarity": "Embedding sim (Phase 5)",
            "unknown": "Unknown",
            "none": "No genre yet",
        }.get(row["source"], row["source"])
        print(f"  {label:<30} {_fmt(row['cnt'], total)}")

    print("\n── Genre Distribution (all sources) ─────────────────────")
    genres = conn.execute(
        """
        SELECT COALESCE(genre_tag, 'Unknown') AS genre, COUNT(*) AS cnt
        FROM songs
        GROUP BY genre
        ORDER BY cnt DESC
        LIMIT 20
        """
    ).fetchall()
    for row in genres:
        print(f"  {row['genre']:<30} {_fmt(row['cnt'], total)}")

    print("\n── Metadata Quality ──────────────────────────────────────")
    quality = conn.execute(
        """
        SELECT COALESCE(metadata_quality, 'none') AS q, COUNT(*) AS cnt
        FROM songs
        GROUP BY q
        ORDER BY cnt DESC
        """
    ).fetchall()
    for row in quality:
        label = {"full": "Full (genre+artist+title)", "partial": "Partial", "none": "None"}.get(
            row["q"], row["q"]
        )
        print(f"  {label:<30} {_fmt(row['cnt'], total)}")

    # Artist/title sources
    a_id3 = conn.execute("SELECT COUNT(*) FROM songs WHERE artist_source='id3'").fetchone()[0]
    a_fn  = conn.execute("SELECT COUNT(*) FROM songs WHERE artist_source='filename'").fetchone()[0]
    a_nil = conn.execute("SELECT COUNT(*) FROM songs WHERE artist_source IS NULL").fetchone()[0]
    t_id3 = conn.execute("SELECT COUNT(*) FROM songs WHERE title_source='id3'").fetchone()[0]
    t_fn  = conn.execute("SELECT COUNT(*) FROM songs WHERE title_source='filename'").fetchone()[0]
    t_nil = conn.execute("SELECT COUNT(*) FROM songs WHERE title_source IS NULL").fetchone()[0]

    print("\n── Artist / Title Sources ────────────────────────────────")
    print(f"  Artist from ID3      {_fmt(a_id3, total)}")
    print(f"  Artist from filename {_fmt(a_fn,  total)}")
    print(f"  Artist missing       {_fmt(a_nil, total)}")
    print(f"  Title  from ID3      {_fmt(t_id3, total)}")
    print(f"  Title  from filename {_fmt(t_fn,  total)}")
    print(f"  Title  missing       {_fmt(t_nil, total)}")

    # Songs that have no usable display name at all
    no_display = conn.execute(
        "SELECT COUNT(*) FROM songs WHERE artist IS NULL AND title IS NULL"
    ).fetchone()[0]
    print(f"\n  No displayable metadata (filename only): {_fmt(no_display, total)}")

    print("\n── Action Items ──────────────────────────────────────────")
    no_genre_count = conn.execute(
        "SELECT COUNT(*) FROM songs WHERE genre_source IS NULL"
    ).fetchone()[0]
    no_genre_pct = 100 * no_genre_count / total if total else 0
    if no_genre_pct > 50:
        print(
            f"  ⚠  {no_genre_pct:.0f}% of songs have no genre — "
            "consider refining FOLDER_GENRE_MAP in config.py before proceeding to Phase 3."
        )
    else:
        print(f"  OK Genre gap: {no_genre_pct:.1f}% — acceptable for YAMNet to fill in Phase 3.")

    if playlisted < 5000:
        print(f"  ⚠  Only {playlisted} playlisted songs found — expected ~8,000. Check playlist paths.")
    else:
        print(f"  OK {playlisted} playlisted songs found.")

    print("=" * 60)
    conn.close()


if __name__ == "__main__":
    run_report()
