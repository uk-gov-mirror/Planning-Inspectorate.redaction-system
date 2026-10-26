#!/usr/bin/env bash
# Keep only sweep rows whose (chunk size, overlap) is in the given grid.
# Usage: filter_sweep.sh IN.jsonl OUT.jsonl "SIZES" "RATIOS"   (run from redactor/)
set -euo pipefail
[[ $# -eq 5 ]] || { echo "usage: $0 IN.jsonl OUT.jsonl \"SIZES\" \"RATIOS\" \"RULES\"" >&2; exit 2; }
[[ -f "$1" ]] || { echo "not found: $1" >&2; exit 1; }
[[ "$1" != "$2" ]] || { echo "OUT must differ from IN" >&2; exit 1; }

# Python computes the overlaps, so rounding matches run_chunk_sweep exactly
PYTHONPATH="$PWD" uv run python - "$@" <<'PY'
import json
import sys
from eval.run_chunk_sweep import overlap_for

src, dst, sizes, ratios, rules = sys.argv[1:6]
keep = {(s, overlap_for(s, r)) for s in map(int, sizes.split()) for r in map(float, ratios.split())}
kept = dropped = 0
with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        if rules and not any(rule in row.get("rule", []) for rule in rules.split()):
            continue
        if (row["chunk_size"], row["chunk_overlap"]) in keep:
            fout.write(line if line.endswith("\n") else line + "\n")
            kept += 1
        else:
            dropped += 1
print(f"kept {kept} rows, removed {dropped}; {len(keep)} variants in the grid -> {dst}")
PY