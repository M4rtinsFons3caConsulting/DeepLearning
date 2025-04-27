"""
Module for reading and updating a CSV file (SINGATURE_FILE) with paths, 
which correspond to either the "CLEANER" or "UPSAMPLER" columns.
The file contains a single row and predefined columns.
"""

import csv
from pathlib import Path
from typing import Dict
from deep.constants import SIGNATURE_FILE, SIGNATURE_COLS, UPSAMPLE_JSONS, CLEANER_JSONS, SPLITTER_JSONS

def write_to(
      path: Path
    ) -> None:
    """
    Updates the appropriate column ("CLEANER" or "UPSAMPLER") in the CSV 
    with the provided path if it belongs to CLEANER_JSONS or UPSAMPLE_JSONS. 
    Raises ValueError if the path is not recognized.
    """

    if path.parent == UPSAMPLE_JSONS:
        col = "UPSAMPLER"
    elif path.parent == CLEANER_JSONS:
        col = "CLEANER"
    elif path.parent == SPLITTER_JSONS:
        col = "SPLITTER"
    else:
        raise ValueError("Provided str is not in UPSAMPLE_JSONS or CLEANER_JSONS.")

    row = {col_name: "" for col_name in SIGNATURE_COLS}

    try:
        with open(SIGNATURE_FILE, newline='') as f:
            reader = csv.DictReader(f)
            row.update(next(reader, row))
    except FileNotFoundError:
        pass

    row[col] = str(path)

    with open(SIGNATURE_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SIGNATURE_COLS)
        writer.writeheader()
        writer.writerow(row)


def write_from() -> Dict[str, str]:
    """
    Reads the single row from the CSV file and returns it as a dictionary.
    """

    with open(SIGNATURE_FILE, newline='') as f:
        reader = csv.DictReader(f)
        return next(reader)
