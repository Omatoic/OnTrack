"""Re-solve helpers: prune bad keypoints/matches from the COLMAP database.

Given a set of bad points3D ids (by reprojection error or manual deletion),
this module maps them back to their 2D observations, dumps the database
keypoints and verified matches to COLMAP text format minus the bad entries,
and rebuilds a fresh database so GLOMAP can be re-run.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

MAX_IMAGE_ID = 2147483647


def pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    return pair_id // MAX_IMAGE_ID, pair_id % MAX_IMAGE_ID


def image_ids_to_pair_id(image_id1: int, image_id2: int) -> int:
    return image_id1 * MAX_IMAGE_ID + image_id2


def _load_images(conn: sqlite3.Connection) -> dict[int, str]:
    return {i: n for i, n in conn.execute("SELECT image_id, name FROM images")}


def _load_keypoints(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    out = {}
    for image_id, rows, cols, data in conn.execute(
        "SELECT image_id, rows, cols, data FROM keypoints"
    ):
        kp = np.frombuffer(data, dtype=np.float32).reshape(rows, cols)
        out[image_id] = kp
    return out


def _load_descriptors(conn: sqlite3.Connection) -> dict[int, np.ndarray]:
    out = {}
    for image_id, rows, cols, data in conn.execute(
        "SELECT image_id, rows, cols, data FROM descriptors"
    ):
        desc = np.frombuffer(data, dtype=np.uint8).reshape(rows, cols)
        out[image_id] = desc
    return out


def _load_two_view_matches(conn: sqlite3.Connection) -> dict[tuple[int, int], np.ndarray]:
    """Load verified inlier matches from two_view_geometries (config >= 2)."""
    out: dict[tuple[int, int], np.ndarray] = {}
    for pair_id, config, data in conn.execute(
        "SELECT pair_id, config, data FROM two_view_geometries"
    ):
        if config < 2 or not data:
            continue
        pairs = np.frombuffer(data, dtype=np.uint32).reshape(-1, 2)
        if len(pairs):
            out[pair_id_to_image_ids(pair_id)] = pairs
    if not out:
        for pair_id, data in conn.execute("SELECT pair_id, data FROM matches"):
            if not data:
                continue
            pairs = np.frombuffer(data, dtype=np.uint32).reshape(-1, 2)
            if len(pairs):
                out[pair_id_to_image_ids(pair_id)] = pairs
    return out


def _write_feature_txt(path: Path, kp: np.ndarray, desc: np.ndarray) -> None:
    """Write COLMAP feature text: header then 'x y scale orient d1..dN'."""
    n = kp.shape[0]
    dim = desc.shape[1] if desc.size else 128
    lines = [f"{n} {dim}"]
    if desc.size == 0:
        desc = np.zeros((n, dim), dtype=np.uint8)
    for i in range(n):
        x, y = kp[i, 0], kp[i, 1]
        # DB keypoints may be stored as affine (a11,a12,a21,a22) which is not a
        # valid scale/orientation pair; write neutral values — only x,y matter
        # for geometric verification of imported matches.
        d = " ".join(str(int(v)) for v in desc[i])
        lines.append(f"{x:.3f} {y:.3f} 1.000 0.000 {d}")
    path.write_text("\n".join(lines) + "\n")


def _write_match_list(path: Path, entries: list[tuple[str, str, np.ndarray]]) -> None:
    with open(path, "w") as f:
        for name_a, name_b, m in entries:
            f.write(f"{name_a} {name_b}\n")
            f.writelines(f"{i} {j}\n" for i, j in m.tolist())
            f.write("\n")


def prune_and_rebuild(
    workspace,
    bad_obs: dict[int, set[int]],
    log=print,
) -> Path:
    """Write pruned features/matches and a fresh database; return db path.

    ``bad_obs`` maps image_id -> set of bad keypoint indices.
    """
    src_db = workspace.database_file
    conn = sqlite3.connect(str(src_db))
    images = _load_images(conn)
    keypoints = _load_keypoints(conn)
    descriptors = _load_descriptors(conn)
    matches = _load_two_view_matches(conn)
    conn.close()

    prune_dir = workspace.database / "pruned"
    if prune_dir.exists():
        shutil.rmtree(prune_dir)
    feat_dir = prune_dir / "features"
    feat_dir.mkdir(parents=True)

    kept_maps: dict[int, dict[int, int]] = {}
    total_kp, kept_kp = 0, 0
    for image_id, name in images.items():
        kp = keypoints.get(image_id)
        if kp is None:
            continue
        bad = bad_obs.get(image_id, set())
        keep_idx = [i for i in range(len(kp)) if i not in bad]
        kept_maps[image_id] = {old: new for new, old in enumerate(keep_idx)}
        total_kp += len(kp)
        kept_kp += len(keep_idx)

        desc = descriptors.get(image_id)
        desc = desc[keep_idx] if desc is not None else np.zeros((0, 0), np.uint8)
        _write_feature_txt(feat_dir / f"{name}.txt", kp[keep_idx], desc)

    removed_pairs = 0
    kept_pairs: list[tuple[str, str, np.ndarray]] = []
    total_matches, kept_matches = 0, 0
    for (i1, i2), pairs in matches.items():
        total_matches += len(pairs)
        map1, map2 = kept_maps.get(i1, {}), kept_maps.get(i2, {})
        new_pairs = []
        for a, b in pairs:
            na, nb = map1.get(int(a)), map2.get(int(b))
            if na is not None and nb is not None:
                new_pairs.append((na, nb))
        kept_matches += len(new_pairs)
        if new_pairs:
            kept_pairs.append((images[i1], images[i2], np.asarray(new_pairs, np.uint32)))
        else:
            removed_pairs += 1

    match_file = prune_dir / "matches.txt"
    _write_match_list(match_file, kept_pairs)

    log(
        f"Pruned {total_kp - kept_kp}/{total_kp} keypoints, "
        f"{total_matches - kept_matches}/{total_matches} match inliers, "
        f"{removed_pairs} pairs emptied."
    )

    new_db = workspace.database / "database_pruned.db"
    if new_db.exists():
        new_db.unlink()
    return new_db, feat_dir, match_file
