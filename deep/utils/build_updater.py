"""
This module manages the current data preprocessing configuration in a user session. It tracks the active cleaning, upsampling, 
and splitting routines by recording the corresponding JSON filenames in a CSV file. This allows users to verify which data 
modifications are applied to the current working set.
"""

import csv
from pathlib import Path
from typing import Dict
from deep.constants import SIGNATURE_FILE, SIGNATURE_COLS, UPSAMPLE_JSONS, CLEANER_JSONS, SPLITTER_JSONS

def write_to(
      path: Path
    ) -> None:
    """
    Updates the appropriate column in the signature CSV.
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

    row[col] = path.name

    with open(SIGNATURE_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SIGNATURE_COLS)
        writer.writeheader()
        writer.writerow(row)


def write_from() -> Dict[str, str]:
    """
    Reads from the signature CSV, file into a dictionary.
    """

    with open(SIGNATURE_FILE, newline='') as f:
        reader = csv.DictReader(f)
        return next(reader)
    
def clean() -> None:
    """
    Clears the contents of the signature CSV, preserving only the headers.
    """
    
    with open(SIGNATURE_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SIGNATURE_COLS)
        writer.writeheader()