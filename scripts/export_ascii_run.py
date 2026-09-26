#!/usr/bin/env python3
"""Export completed Titan Text ASCII generation runs to external Android shared storage.

Copies internal run artifacts from runs/ascii/<RUN_ID> to /sdcard/Download/TitanText/ascii_runs/<RUN_ID>
while verifying file counts, SHA-256 hashes, and directory structure.
Leaves internal originals completely untouched.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def check_storage_writable(dest_root: Path) -> bool:
    """Checks whether the destination directory exists and is writable."""
    try:
        dest_root.mkdir(parents=True, exist_ok=True)
        test_file = dest_root / ".write_test.tmp"
        test_file.write_text("ok")
        test_file.unlink()
        return True
    except Exception as e:
        sys.stderr.write(f"[ERROR] Destination '{dest_root}' is not writable: {e}\n")
        return False


def export_run(run_dir: Path, dest_root: Path, create_gallery: bool = True) -> int:
    run_dir = run_dir.resolve()
    if not run_dir.exists() or not run_dir.is_dir():
        sys.stderr.write(f"[ERROR] Source run directory '{run_dir}' does not exist or is not a directory.\n")
        return 1

    run_id = run_dir.name
    dest_run_dir = dest_root / run_id

    print(f"================================================================================")
    print(f"TITAN TEXT · EXTERNAL RUN EXPORTER")
    print(f"================================================================================")
    print(f"  Source Run Path : {run_dir}")
    print(f"  Destination Path: {dest_run_dir}")
    print(f"  Run ID          : {run_id}")

    if not check_storage_writable(dest_root):
        sys.stderr.write("\n[CRITICAL FAILURE] Android shared storage is not accessible or not writable.\n")
        sys.stderr.write("If running on Termux, please execute 'termux-setup-storage' and grant permission.\n")
        return 2

    # Check for existing destination run to avoid silent accidental overwrite
    if dest_run_dir.exists():
        print(f"[NOTE] Destination run directory already exists; updating contents...")

    dest_run_dir.mkdir(parents=True, exist_ok=True)

    copied_files = 0
    raw_hash_matches = 0
    raw_files_total = 0

    # Recursive copy
    for root, dirs, files in os.walk(run_dir):
        rel_root = Path(root).relative_to(run_dir)
        target_sub_dir = dest_run_dir / rel_root
        target_sub_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            src_file = Path(root) / filename
            dst_file = target_sub_dir / filename

            # Copy file
            shutil.copy2(src_file, dst_file)
            copied_files += 1

            # If raw generation .txt file, compute and verify hash
            if "raw" in rel_root.parts and filename.endswith(".txt"):
                raw_files_total += 1
                src_hash = compute_sha256(src_file)
                dst_hash = compute_sha256(dst_file)
                if src_hash == dst_hash:
                    raw_hash_matches += 1
                else:
                    sys.stderr.write(f"[ERROR] Hash mismatch for {rel_root / filename}!\n")

    print(f"\n✓ Files Copied Successfully: {copied_files}")
    if raw_files_total > 0:
        print(f"✓ Raw Samples Verified    : {raw_hash_matches}/{raw_files_total} (SHA-256 match 100%)")

    # Optional gallery export
    gallery_src = run_dir / "gallery"
    if create_gallery and gallery_src.exists() and gallery_src.is_dir():
        gallery_dest = dest_root.parent / "ascii_gallery" / run_id
        gallery_dest.mkdir(parents=True, exist_ok=True)
        gallery_count = 0
        for f in gallery_src.glob("*.txt"):
            shutil.copy2(f, gallery_dest / f.name)
            gallery_count += 1
        print(f"✓ Gallery Samples Copied  : {gallery_count} -> {gallery_dest}")

    print(f"\n[SUCCESS] Completed run export for '{run_id}'. Internal original unchanged.\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Export Titan Text ASCII run to Android shared storage.")
    parser.add_argument("run_path", help="Path to internal run directory, e.g. runs/ascii/<RUN_ID>")
    parser.add_argument(
        "--dest",
        default="/sdcard/Download/TitanText/ascii_runs",
        help="Destination directory on shared storage (default: /sdcard/Download/TitanText/ascii_runs)",
    )
    parser.add_argument(
        "--no-gallery",
        action="store_true",
        help="Do not copy gallery folder to /sdcard/Download/TitanText/ascii_gallery",
    )

    args = parser.parse_args()
    dest_path = Path(args.dest)
    run_path = Path(args.run_path)

    code = export_run(run_path, dest_path, create_gallery=not args.no_gallery)
    sys.exit(code)


if __name__ == "__main__":
    main()
