"""
This module is responsible for verifying, extracting, and moving files in the unpacking process.

Functions include:
- _calculate_sha256: Calculates the SHA-256 hash of a file.
- _unpack: Unpacks a ZIP file from origin to destination.
- _move_file: Moves a file from the source path to the destination path.
- _verify_zips: Verifies the integrity of ZIP files based on pre-computed checksums.
- _unpack_all: Unpacks all specified ZIP files to their respective destinations.
- _move_all: Moves files based on predefined instructions.
- extract_archives: High-level function that runs the unpacking and file-moving processes.
"""

import csv
import hashlib
import zipfile
import shutil
from pathlib import Path
from deep.constants import CHECKSUM_FILE, ZIP_FILE_INSTRUCTIONS, MOVE_FILE_INSTRUCTIONS


def _calculate_sha256(file_path: Path) -> str:
    """
    Calculates the SHA-256 hash of the given file.

    """
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _unpack(origin: Path, destination: Path) -> None:
    """
    Unpacks a ZIP file from the origin path to the destination path.

    """
    with zipfile.ZipFile(origin) as archive:
        archive.extractall(destination)


def _move_file(
          src: Path
        , dst: Path
        , label: str
    ) -> None:
    """
    Moves a file from the source path to the destination path, and logs the result.

    """
    if src.exists():
        shutil.move(src, dst)
        print(f"{label} moved. From {src} to {dst}")
    else:
        print(f"{src.name} not found at {src}")


def _verify_zips() -> None:
    """
    Verifies the integrity of all specified ZIP files by comparing their current checksum
    with pre-computed expected checksums. Writes a new checksum file if none exists.

    """
    if not CHECKSUM_FILE.exists():
        with open(CHECKSUM_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "checksum"])
            for _, (origin, _, check) in ZIP_FILE_INSTRUCTIONS.items():
                if check:
                    checksum = _calculate_sha256(origin)
                    writer.writerow([origin.name, checksum])
                    print(f"Wrote checksum for {origin.name}: {checksum}")
        print(f"Checksums written to {CHECKSUM_FILE}. Please verify file contents.")
    else:
        with open(CHECKSUM_FILE, "r", newline="") as f:
            reader = csv.DictReader(f)
            expected_checksums = {row["filename"]: row["checksum"] for row in reader}

        for _, (origin, _, check) in ZIP_FILE_INSTRUCTIONS.items():
            if check:
                current_checksum = _calculate_sha256(origin)
                expected_checksum = expected_checksums.get(origin.name)
                if expected_checksum is None:
                    raise RuntimeError(f"No expected checksum for {origin.name}")
                if current_checksum != expected_checksum:
                    raise RuntimeError(f"Checksum mismatch for {origin.name}")
                print(f"{origin.name} passed checksum test.")


def _unpack_all() -> None:
    """
    Unpacks all ZIP files specified in `ZIP_FILE_INSTRUCTIONS` to their respective destinations.

    """
    for _, (origin, destination, _) in ZIP_FILE_INSTRUCTIONS.items():
        try:
            print(f"Extracting {origin.name} to {destination}...")
            _unpack(origin, destination)
            print(f"{origin.name} extraction complete.")
        except Exception as e:
            print(f"Failed to extract {origin.name}: {e}")


def _move_all() -> None:
    """
    Moves files based on the instructions provided in `MOVE_FILE_INSTRUCTIONS`.
    Uses the `_move_file` function to move files to their designated destinations.

    """
    for label, (src, dst) in MOVE_FILE_INSTRUCTIONS.items():
        _move_file(src, dst, label)

def extract_archives() -> None:
    """
    High-level function that runs the entire process of verifying ZIP files, unpacking them,
    and moving the necessary files.

    """
    _verify_zips()
    _unpack_all()
    _move_all()
    print("Resource unpacking complete.")
