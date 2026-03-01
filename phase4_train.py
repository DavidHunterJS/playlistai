"""
Phase 4: Model Training + Genre Centroid Computation

Trains a binary "David likes this" classifier on 1024-dim YAMNet embeddings
extracted in Phase 3. Then computes per-genre mean embeddings (centroids)
from the playlisted training songs and stores them in the DB for the
Layer-4 genre fallback used in Phase 5.

Runs on RunPod after phase3_extract.py completes.

Setup (already done if Phase 3 ran):
    pip install tensorflow numpy pandas tqdm

Usage:
    python phase4_train.py [--model-dir ./model] [--epochs 200] [--batch-size 256]

Outputs (written to model_dir):
    playlistai_model.keras  — trained Keras model
    norm_stats.npz          — training-set mean + std for inference normalisation
    model_version.txt       — timestamp string used as model_version in predictions table
"""

import os
import logging
import argparse
import datetime
import collections
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import DB_PATH, RANDOM_SEED
from db import init_db, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("phase4_train.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

EMBEDDING_DIM = 1024


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_dataset(conn):
    """
    Pull embeddings + labels from DB.
    Returns (song_ids, embeddings, labels, splits, genres) as numpy arrays /
    Python lists.  Only rows with a non-NULL embedding are returned.
    """
    total_manifest = conn.execute(
        "SELECT COUNT(*) FROM training_manifest"
    ).fetchone()[0]

    rows = conn.execute(
        """
        SELECT tm.song_id, tm.label, tm.split, s.embedding, s.genre_tag
        FROM training_manifest tm
        JOIN songs s ON s.song_id = tm.song_id
        WHERE s.embedding IS NOT NULL
        ORDER BY tm.song_id
        """
    ).fetchall()

    if not rows:
        raise RuntimeError(
            f"No embeddings found — run phase3_extract.py first. "
            f"({total_manifest} songs in training_manifest, 0 have embeddings)"
        )

    missing = total_manifest - len(rows)
    if missing > 0:
        log.warning(
            "%d / %d training songs are missing embeddings (%.1f%%) — "
            "continuing with %d songs; run phase3_extract.py to fill gaps",
            missing, total_manifest, 100 * missing / total_manifest, len(rows),
        )
    else:
        log.info("All %d training songs have embeddings.", len(rows))

    song_ids  = np.array([r["song_id"] for r in rows], dtype=np.int64)
    embeddings = np.stack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    labels = np.array([r["label"] for r in rows], dtype=np.float32)
    splits = [r["split"] for r in rows]
    genres = [r["genre_tag"] for r in rows]

    return song_ids, embeddings, labels, splits, genres


# ── Model ──────────────────────────────────────────────────────────────────────

def _build_model() -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(EMBEDDING_DIM,), name="embedding"),
        tf.keras.layers.Dense(256, activation="relu", name="dense_1"),
        tf.keras.layers.Dropout(0.5, name="dropout_1"),
        tf.keras.layers.Dense(128, activation="relu", name="dense_2"),
        tf.keras.layers.Dropout(0.5, name="dropout_2"),
        tf.keras.layers.Dense(1, activation="sigmoid", name="output"),
    ], name="playlistai")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# ── Genre centroids ────────────────────────────────────────────────────────────

def _compute_centroids(embeddings, labels, genres):
    """
    Compute mean raw embedding per genre from positive training examples.
    Returns dict: genre_tag -> (centroid: np.ndarray float32, count: int)
    Centroids are stored raw (pre-normalisation) so Phase 5 can compute
    cosine similarity against raw song embeddings.
    """
    buckets = collections.defaultdict(list)
    for i, (label, genre) in enumerate(zip(labels, genres)):
        if label == 1 and genre is not None:
            buckets[genre].append(embeddings[i])

    return {
        genre: (np.stack(embs).mean(axis=0).astype(np.float32), len(embs))
        for genre, embs in buckets.items()
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def run_training(
    db_path: str = DB_PATH,
    model_dir: str = "model",
    epochs: int = 200,
    batch_size: int = 256,
):
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    init_db(db_path)
    conn = get_connection(db_path)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model_version = datetime.datetime.utcnow().strftime("v%Y%m%d_%H%M")
    log.info("Model version: %s", model_version)

    # ── Load ───────────────────────────────────────────────────────────────────
    log.info("Loading embeddings from DB …")
    song_ids, embeddings, labels, splits, genres = _load_dataset(conn)

    train_mask = np.array([s == "train" for s in splits])
    val_mask   = ~train_mask

    X_train, y_train = embeddings[train_mask], labels[train_mask]
    X_val,   y_val   = embeddings[val_mask],   labels[val_mask]

    log.info(
        "Train: %d  (%d pos / %d neg)   Val: %d  (%d pos / %d neg)",
        len(X_train), int(y_train.sum()), int((1 - y_train).sum()),
        len(X_val),   int(y_val.sum()),   int((1 - y_val).sum()),
    )

    # ── Normalise (training-set statistics only) ───────────────────────────────
    norm_mean = X_train.mean(axis=0)
    norm_std  = X_train.std(axis=0) + 1e-8
    X_train_n = (X_train - norm_mean) / norm_std
    X_val_n   = (X_val   - norm_mean) / norm_std

    norm_path = os.path.join(model_dir, "norm_stats.npz")
    np.savez(norm_path, mean=norm_mean, std=norm_std)
    log.info("Norm stats saved → %s", norm_path)

    # ── Build ──────────────────────────────────────────────────────────────────
    log.info("Building model …")
    model = _build_model()
    model.summary(print_fn=log.info)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=15,
            restore_best_weights=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "checkpoint.keras"),
            save_best_only=True,
            monitor="val_accuracy",
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ── Train ──────────────────────────────────────────────────────────────────
    log.info("Training — max %d epochs, batch size %d …", epochs, batch_size)
    history = model.fit(
        X_train_n, y_train,
        validation_data=(X_val_n, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    val_results = model.evaluate(X_val_n, y_val, verbose=0)
    val_metrics = dict(zip(model.metrics_names, val_results))

    log.info(
        "Final val — loss: %.4f  acc: %.4f  AUC: %.4f  P: %.4f  R: %.4f",
        val_metrics["loss"], val_metrics["accuracy"], val_metrics["auc"],
        val_metrics["precision"], val_metrics["recall"],
    )

    if val_metrics["accuracy"] < 0.75:
        log.warning(
            "Val accuracy %.1f%% is below the 75%% target — "
            "consider more training data or hyperparameter tuning.",
            100 * val_metrics["accuracy"],
        )

    # ── Save model ─────────────────────────────────────────────────────────────
    model_path = os.path.join(model_dir, "playlistai_model.keras")
    model.save(model_path)
    log.info("Model saved → %s", model_path)

    version_path = os.path.join(model_dir, "model_version.txt")
    Path(version_path).write_text(model_version)

    # ── Genre centroids ────────────────────────────────────────────────────────
    log.info("Computing genre centroids from training positives …")
    centroid_data = _compute_centroids(embeddings, labels, genres)

    if centroid_data:
        conn.execute("DELETE FROM genre_centroids")
        conn.executemany(
            "INSERT INTO genre_centroids (genre, centroid, song_count) VALUES (?, ?, ?)",
            [
                (genre, centroid.tobytes(), count)
                for genre, (centroid, count) in centroid_data.items()
            ],
        )
        conn.commit()
        log.info(
            "Genre centroids written for %d genres: %s",
            len(centroid_data),
            ", ".join(f"{g}({c})" for g, (_, c) in centroid_data.items()),
        )
    else:
        log.warning(
            "No genre centroids computed — no positive training songs have genre_tag set. "
            "Layer-4 genre fallback will be unavailable in Phase 5."
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    best_epoch = int(np.argmax(history.history["val_accuracy"])) + 1

    print("\n" + "=" * 60)
    print("PLAYLISTAI — PHASE 4 TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Model version    :  {model_version}")
    print(f"  Training songs   :  {len(X_train):>8,}  ({int(y_train.sum()):,} pos / {int((1-y_train).sum()):,} neg)")
    print(f"  Val songs        :  {len(X_val):>8,}  ({int(y_val.sum()):,} pos / {int((1-y_val).sum()):,} neg)")
    print(f"  Epochs trained   :  {len(history.epoch):>8,}  (best: {best_epoch})")
    print(f"  Val accuracy     :  {val_metrics['accuracy']:>11.4f}  ({100*val_metrics['accuracy']:.1f}%)")
    print(f"  Val AUC          :  {val_metrics['auc']:>11.4f}")
    print(f"  Val precision    :  {val_metrics['precision']:>11.4f}")
    print(f"  Val recall       :  {val_metrics['recall']:>11.4f}")
    print(f"  Genre centroids  :  {len(centroid_data):>8,}")
    print(f"  Model saved      :  {model_path}")
    print(f"  Norm stats       :  {norm_path}")
    print("=" * 60)
    if val_metrics["accuracy"] >= 0.75:
        print("  Target accuracy (>=75%) ACHIEVED ✓")
    else:
        print("  Below target accuracy (75%) — consider more data or tuning")
    print()

    log.info("Phase 4 complete.")
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PlaylistAI Phase 4 model training")
    parser.add_argument(
        "--db-path", default=DB_PATH,
        help="Path to SQLite database (default: %(default)s)",
    )
    parser.add_argument(
        "--model-dir", default="model",
        help="Directory to write model + norm stats (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Maximum training epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Mini-batch size (default: %(default)s)",
    )
    args = parser.parse_args()

    run_training(
        db_path=args.db_path,
        model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
