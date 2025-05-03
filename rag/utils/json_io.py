import json, pathlib
from typing import List, Dict

def read_jsonl(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]

def write_jsonl(data: List[Dict], path: str) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for obj in data:
            json.dump(obj, f)
            f.write("\n")
