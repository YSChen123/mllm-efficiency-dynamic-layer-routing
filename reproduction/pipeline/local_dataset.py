"""Load the exact prepared Arrow files without Hub resolution or data copies."""
import json
from pathlib import Path
from datasets import Dataset, DatasetDict, concatenate_datasets


def load_prepared_dataset(manifest):
    result = {}
    for split, files in manifest['arrow_files'].items():
        if not files:
            raise ValueError(f'No Arrow files for {split}')
        parts = [Dataset.from_file(filename) for filename in files]
        data = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        if len(data) != manifest['splits'][split]:
            raise ValueError(f'Prepared Arrow sample count mismatch for {split}')
        result[split] = data
    return DatasetDict(result)
