"""Minimal GCS prefix sync (one-way): download (default) or upload.

Remote -> Local (download):
    python gcp_bucket_watcher.py --bucket ctx-to-lora --prefix train_outputs --dest train_outputs
Local  -> Remote (upload):
    python gcp_bucket_watcher.py --bucket ctx-to-lora --prefix train_outputs --dest train_outputs --upload

Mirrors only new / changed files (size or mtime difference). No deletions.
Filtering: --include / --exclude accept comma separated globs. Auth uses
Application Default Credentials or --credentials JSON.

Example
# download
python gcp_bucket_watcher.py --bucket ctx-to-lora --prefix train_outputs --dest train_outputs
# upload
python gcp_bucket_watcher.py --bucket ctx-to-lora --prefix train_outputs --dest train_outputs --upload
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

try:
    from google.cloud import storage  # type: ignore
    from google.oauth2 import service_account  # type: ignore
except Exception as e:  # pragma: no cover - import guard
    print("google-cloud-storage not installed. Add to dependencies.", file=sys.stderr)
    raise


@dataclass
class ObjectMeta:  # Unified meta for local + remote
    name: str  # relative path (no leading prefix for remote)
    updated: float  # mtime (remote object updated timestamp or local mtime)
    size: int  # bytes


def parse_time_string(time_str: str) -> float:
    """Parse human-readable time string to timestamp.

    Supports formats:
    - ISO format: 2024-01-15T10:30:00 or 2024-01-15 10:30:00
    - Date only: 2024-01-15 (assumes 00:00:00)
    - Unix timestamp: 1705315800
    """
    # Try parsing as unix timestamp first
    try:
        return float(time_str)
    except ValueError:
        pass

    # Try various datetime formats
    formats = [
        "%Y-%m-%dT%H:%M:%S",  # 2024-01-15T10:30:00
        "%Y-%m-%d %H:%M:%S",  # 2024-01-15 10:30:00
        "%Y-%m-%dT%H:%M",  # 2024-01-15T10:30
        "%Y-%m-%d %H:%M",  # 2024-01-15 10:30
        "%Y-%m-%d",  # 2024-01-15
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(time_str, fmt)
            return dt.timestamp()
        except ValueError:
            continue

    raise ValueError(f"Unable to parse time string: {time_str}")


def parse_globs(spec: str | None) -> list[str]:
    if not spec:
        return []
    return [g.strip() for g in spec.split(",") if g.strip()]


def should_skip(
    rel_path: str,
    includes: Sequence[str],
    excludes: Sequence[str],
    meta: ObjectMeta | None = None,
    from_time: float | None = None,
) -> bool:
    base = os.path.basename(rel_path)
    if excludes and any(fnmatch.fnmatch(base, p) for p in excludes):
        return True
    if includes and not any(fnmatch.fnmatch(base, p) for p in includes):
        return True
    if from_time is not None and meta is not None and meta.updated < from_time:
        return True
    return False


def list_remote(
    client: storage.Client, bucket: str, prefix: str
) -> dict[str, ObjectMeta]:
    out: dict[str, ObjectMeta] = {}
    p = prefix.rstrip("/")
    for b in client.list_blobs(bucket_or_name=bucket, prefix=prefix):
        if b.name.endswith("/"):
            continue
        rel = b.name[len(p) :].lstrip("/") if p else b.name
        out[rel] = ObjectMeta(
            rel, (b.updated.timestamp() if b.updated else 0.0), b.size or 0
        )
    return out


def list_local(root: str) -> dict[str, ObjectMeta]:
    out: dict[str, ObjectMeta] = {}
    for base, _dirs, files in os.walk(root):
        for f in files:
            path = os.path.join(base, f)
            try:
                st = os.stat(path)
            except OSError:
                continue
            rel = os.path.relpath(path, root).replace("\\", "/")
            out[rel] = ObjectMeta(rel, st.st_mtime, st.st_size)
    return out


def needs_transfer(src: ObjectMeta, dst: ObjectMeta | None, tolerance: float) -> bool:
    if dst is None:
        return True
    if src.size != dst.size:
        return True
    if abs(src.updated - dst.updated) > tolerance:
        return True
    return False


# --- Minimal persistent state to honor local deletions ---
def _state_path(dest: str) -> str:
    return os.path.join(dest, ".gcs_sync_state.json")


def _load_state(dest: str) -> tuple[set[str], set[str]]:
    path = _state_path(dest)
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("downloaded", [])), set(data.get("tombstones", []))
    except FileNotFoundError:
        return set(), set()
    except Exception:
        # Corrupt/invalid state; ignore to keep going.
        return set(), set()


def _save_state(dest: str, downloaded: set[str], tombstones: set[str]) -> None:
    path = _state_path(dest)
    tmp = path + ".part"
    data = {
        "downloaded": sorted(downloaded),
        "tombstones": sorted(tombstones),
    }
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def download_file(
    client: storage.Client, bucket: str, prefix: str, meta: ObjectMeta, dest: str
) -> None:
    blob_name = f"{prefix.rstrip('/')}/{meta.name}" if prefix else meta.name
    blob = client.bucket(bucket).blob(blob_name)
    local_path = os.path.join(dest, meta.name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp = local_path + ".part"
    blob.download_to_filename(tmp)
    os.utime(tmp, (time.time(), meta.updated))
    os.replace(tmp, local_path)


def upload_file(
    client: storage.Client, bucket: str, prefix: str, meta: ObjectMeta, src_root: str
) -> None:
    blob_name = f"{prefix.rstrip('/')}/{meta.name}" if prefix else meta.name
    blob = client.bucket(bucket).blob(blob_name)
    blob.upload_from_filename(os.path.join(src_root, meta.name))


def sync_cycle(args: argparse.Namespace, client: storage.Client) -> int:
    includes, excludes = parse_globs(args.include), parse_globs(args.exclude)
    remote = list_remote(client, args.bucket, args.prefix)
    local = list_local(args.dest)
    changed = 0
    if args.upload:  # local -> remote (original tolerance 2s)
        for rel, meta in local.items():
            if should_skip(rel, includes, excludes, meta, args.from_time):
                continue
            remote_meta = remote.get(rel)
            # For uploads we consider a file up-to-date if sizes match and the remote
            # object timestamp is >= local mtime (within tolerance). Using absolute
            # delta caused perpetual re-uploads because GCS sets its own server time.
            if (
                remote_meta
                and meta.size == remote_meta.size
                and remote_meta.updated >= meta.updated - 2.0
            ):
                continue
            if needs_transfer(meta, remote_meta, tolerance=2.0):
                print(f"[uploading ] {rel} ({meta.size} bytes)")
                try:
                    upload_file(client, args.bucket, args.prefix, meta, args.dest)
                except FileNotFoundError:
                    # File disappeared between scan and upload attempt; skip gracefully.
                    print(f"[skipped missing] {rel}")
                    continue
                print(f"[uploaded  ✓] {rel} ({meta.size} bytes)")
                changed += 1
        if changed == 0:
            print("No new or updated local files to upload.")
        else:
            print("All uploads complete.")
    else:  # download (original tolerance 1s)
        downloaded_set, tombstones = _load_state(args.dest)
        state_dirty = False
        for rel, meta in remote.items():
            if should_skip(rel, includes, excludes, meta, args.from_time):
                continue
            # Respect user-deleted files (tombstones)
            if rel in tombstones:
                continue
            # If we had downloaded this before but it's now missing locally,
            # treat as user-deleted and tombstone it to avoid re-download.
            if rel in downloaded_set and rel not in local:
                tombstones.add(rel)
                state_dirty = True
                continue
            local_meta = local.get(rel)

            # Skip if local file is newer than remote (avoid overwriting newer local files)
            if local_meta and local_meta.updated > meta.updated + 1.0:  # 1s tolerance
                if rel not in downloaded_set:
                    downloaded_set.add(rel)
                    state_dirty = True
                continue

            if needs_transfer(meta, local_meta, tolerance=1.0):
                print(f"[downloading] {rel} ({meta.size} bytes)")
                download_file(client, args.bucket, args.prefix, meta, args.dest)
                print(f"[downloaded ✓] {rel} ({meta.size} bytes)")
                changed += 1
                if rel not in downloaded_set:
                    downloaded_set.add(rel)
                    state_dirty = True
                if rel in tombstones:
                    tombstones.discard(rel)
                    state_dirty = True
            else:
                # File exists locally and is in sync; ensure it's marked as downloaded
                # so future local deletions are respected without re-download.
                if local_meta is not None and rel not in downloaded_set:
                    downloaded_set.add(rel)
                    state_dirty = True
        if changed == 0:
            print("No new or updated objects.")
        else:
            print("All downloads complete.")
        if state_dirty:
            _save_state(args.dest, downloaded_set, tombstones)
    return changed


def build_client(args: argparse.Namespace) -> storage.Client:
    if args.credentials:
        creds = service_account.Credentials.from_service_account_file(args.credentials)
        return storage.Client(credentials=creds, project=creds.project_id)
    return storage.Client()  # ADC


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch and mirror a GCS prefix locally.")
    p.add_argument("--bucket", required=True, help="GCS bucket name (without gs://)")
    p.add_argument(
        "--prefix",
        default="",
        help="Prefix inside the bucket to mirror (e.g. train_outputs)",
    )
    p.add_argument(
        "--dest", default="train_outputs", help="Local destination directory"
    )
    p.add_argument(
        "--interval", type=int, default=60, help="Seconds between scans (default 60)"
    )
    p.add_argument("--once", action="store_true", help="Run a single scan and exit")
    p.add_argument(
        "--upload",
        action="store_true",
        help="Reverse direction: upload local files to bucket (default is download).",
    )
    p.add_argument("--include", help="Comma separated glob(s) to include (default all)")
    p.add_argument("--exclude", help="Comma separated glob(s) to exclude")
    p.add_argument(
        "--from-time",
        help="Only sync files modified after this time (e.g. '2024-01-15', '2024-01-15T10:30:00', or unix timestamp)",
    )
    p.add_argument("--credentials", help="Path to service account JSON (optional)")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    # Parse from_time if provided
    if args.from_time:
        try:
            args.from_time = parse_time_string(args.from_time)
        except ValueError as e:
            print(f"Error parsing --from-time: {e}", file=sys.stderr)
            return 1

    client = build_client(args)
    os.makedirs(args.dest, exist_ok=True)
    try:
        while True:
            print(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                "Scanning (upload)" if args.upload else "Scanning (download)",
            )
            sync_cycle(args, client)
            if args.once:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Interrupted.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
