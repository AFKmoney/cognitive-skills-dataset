#!/usr/bin/env python3
"""Build an index.jsonl listing every file with its entry count and size."""
import os, json

DATA = "/home/z/my-project/cognitive-skills/data"
out_path = os.path.join(DATA, "index.jsonl")

entries = []
total_entries = 0
total_bytes = 0
total_tokens = 0

for fn in sorted(os.listdir(DATA)):
    if not fn.endswith(".jsonl") or fn == "index.jsonl":
        continue
    path = os.path.join(DATA, fn)
    size = os.path.getsize(path)
    n = 0
    chars = 0
    with open(path) as f:
        for line in f:
            n += 1
            e = json.loads(line)
            for m in e["messages"]:
                chars += len(m["content"])
    tokens = chars // 4
    skill, subtopic = fn.replace(".jsonl", "").split("__", 1)
    entries.append({
        "file": fn,
        "skill": skill,
        "subtopic": subtopic,
        "entries": n,
        "bytes": size,
        "approx_tokens": tokens,
    })
    total_entries += n
    total_bytes += size
    total_tokens += tokens

# Sort by skill then subtopic
order = {"thinking": 0, "reasoning": 1, "speaking": 2, "understanding": 3, "coding": 4}
entries.sort(key=lambda e: (order.get(e["skill"], 99), e["subtopic"]))

with open(out_path, "w") as f:
    for e in entries:
        f.write(json.dumps(e) + "\n")

print(f"Wrote {out_path}")
print(f"Total files: {len(entries)}")
print(f"Total entries: {total_entries:,}")
print(f"Total bytes: {total_bytes:,}")
print(f"Approx total tokens: {total_tokens:,}")

# Also print summary for the README
print("\n--- Per-skill summary ---")
by_skill = {}
for e in entries:
    s = e["skill"]
    if s not in by_skill:
        by_skill[s] = {"files": 0, "entries": 0, "tokens": 0, "bytes": 0}
    by_skill[s]["files"] += 1
    by_skill[s]["entries"] += e["entries"]
    by_skill[s]["tokens"] += e["approx_tokens"]
    by_skill[s]["bytes"] += e["bytes"]

for skill in ["thinking", "reasoning", "speaking", "understanding", "coding"]:
    s = by_skill[skill]
    print(f"{skill:14s}  files={s['files']:3d}  entries={s['entries']:>9,}  tokens={s['tokens']:>12,}  size={s['bytes']/1024/1024:>8.1f} MB")
