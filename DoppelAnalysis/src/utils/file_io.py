import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Union

def grab_pair_id(voter_id: str) -> int:
    return int(voter_id.split("_")[1])

def load_jsonl(path: Path) -> List[dict]:
    """Load a JSONL file as a list of dictionaries."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_json(path: Path) -> dict:
    """Load a JSON file as a dictionary."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def load_csv(path: Path, sep: str = ",") -> List[dict]:
    """Loads a CSV file into a list of dictionaries with a custom separator."""
    with path.open(encoding="utf-8") as fi:
        # DictReader uses 'delimiter' as the parameter name
        reader = csv.DictReader(fi, delimiter=sep)
        data = list(reader)
    return data

def dump_list_to_jsonl(list_obj: List[Union[Dict, str]], out_path: Path) -> None:
    """Write a list of dictionaries or strings to a JSONL file."""
    with out_path.open("w", encoding="utf-8") as f:
        for item in list_obj:
            f.write(json.dumps(item) + "\n")

def append_jsonl(output_path: str, data: dict):
    """
    Append a dictionary to the specified output JSONL file.
    Creates parent directories if needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "a", encoding="utf-8") as f:
        json.dump(data, f)
        f.write('\n')

import os
import csv

def append_tsv(output_path: str, data: dict):
    """
    Append a dictionary to the specified output TSV file.
    Creates parent directories if needed, and validates schema consistency.
    """
    # 1. Create parent directories if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    file_exists = os.path.isfile(output_path) and os.path.getsize(output_path) > 0
    input_headers = list(data.keys())

    # 2 & 3. If file exists, check content and schema consistency
    if file_exists:
        with open(output_path, "r", encoding="utf-8") as f:
            first_line = f.readline().rstrip("\n")
            existing_headers = first_line.split("\t")
            
        # Check if headers and column count match exactly (including order)
        if existing_headers != input_headers:
            raise ValueError(
                f"TSV Schema mismatch for '{output_path}'!\n"
                f"Existing headers: {existing_headers} ({len(existing_headers)} cols)\n"
                f"Incoming headers: {input_headers} ({len(input_headers)} cols)"
            )

    # 4. Append data (and write headers if it's a new file)
    with open(output_path, "a", encoding="utf-8", newline="") as f:
        # Using csv.DictWriter ensures values with tabs/newlines are safely quoted
        writer = csv.DictWriter(f, fieldnames=input_headers, delimiter="\t")
        
        if not file_exists:
            writer.writeheader()
            
        writer.writerow(data)