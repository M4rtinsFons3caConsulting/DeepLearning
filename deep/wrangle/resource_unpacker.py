"""
Verifies the integrity of .zip files in `/resources` using pre-computed checksums.
If checks are successful, data is extracted to designated destination folders.
"""

import csv
import hashlib
import zipfile
from pathlib import Path
from deep.constants import ZIP_FILE_INSTRUCTIONS, CHECKSUM_FILE


def _calculate_sha256(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _unpack(origin: Path, destination: Path) -> None:
    with zipfile.ZipFile(origin) as archive:
        archive.extractall(destination)


def _verify_zips() -> None:
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
    for _, (origin, destination, _) in ZIP_FILE_INSTRUCTIONS.items():
        try:
            print(f"Extracting {origin.name} to {destination}...")
            _unpack(origin, destination)
            print(f"{origin.name} extraction complete.")
        except Exception as e:
            print(f"Failed to extract {origin.name}: {e}")


def extract_archives() -> None:
    _verify_zips()
    _unpack_all()
