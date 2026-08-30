"""Inspect public_set.jsonl before committing to an architecture.

Run from the repo root:
    python3 scripts/inspect_dataset.py [path/to/public_set.jsonl]

Answers three questions:
  1. Do samples carry intent_card / behavior, or are they generated at runtime?
  2. What is the scenario mix?
  3. What is actually inside user_profile (passed to your agent.reset)?
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DATASET = sys.argv[1] if len(sys.argv) > 1 else "data/public_set.jsonl"

path = Path(DATASET)
if not path.exists():
    raise SystemExit(f"not found: {path}  (run from the repo root)")

samples = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
print(f"sessions: {len(samples)}\n")

# keys present
key_counts: Counter[str] = Counter()
for sample in samples:
    key_counts.update(sample.keys())

print("top-level keys (sessions containing each):")
for key, count in key_counts.most_common():
    print(f"  {key:20s} {count}")

# gating question
print("\n=== GATING QUESTION ===")
with_card = sum("intent_card" in s for s in samples)
with_behavior = sum("behavior" in s for s in samples)
print(f"  samples carrying intent_card: {with_card}/{len(samples)}")
print(f"  samples carrying behavior:    {with_behavior}/{len(samples)}")

if with_card == 0:
    print("  -> Cards are built from the catalog at runtime by materialize_hidden_fields().")
    print("     Offline inversion of intent_card() is safe to build on.")
elif with_card == len(samples):
    print("  -> Cards ship with the data. The private set probably ships its own too,")
    print("     and their text may not match catalog entries. Use fuzzy matching,")
    print("     not exact-string dictionary lookup.")
else:
    print("  -> Mixed. Your agent must handle both paths.")

# scenario mix
print("\nscenario distribution:")
scenarios = Counter(s.get("scenario_type", "<missing>") for s in samples)
for name, count in scenarios.most_common():
    print(f"  {name:18s} {count:4d}  ({count / len(samples):.0%})")

# ground truth
missing_gt = [s.get("sample_id") for s in samples if "ground_truth" not in s]
if missing_gt:
    print(f"\nWARNING: {len(missing_gt)} samples lack ground_truth")

# user_profile
print("\nuser_profile keys across all sessions:")
profile_keys: Counter[str] = Counter()
non_dict = 0
for sample in samples:
    profile = sample.get("user_profile")
    if isinstance(profile, dict):
        profile_keys.update(profile.keys())
    else:
        non_dict += 1
for key, count in profile_keys.most_common():
    print(f"  {key:24s} {count}")
if non_dict:
    print(f"  ({non_dict} sessions have a non-dict user_profile)")

print("\nfirst 3 user_profile values:")
for sample in samples[:3]:
    print("  " + json.dumps(sample.get("user_profile"))[:400])

# intent_card shape if there is any
if with_card:
    print("\nfirst 3 intent_card values:")
    for sample in samples[:3]:
        if "intent_card" in sample:
            print("  " + json.dumps(sample["intent_card"])[:400])

# full example
print("\n=== FULL FIRST SAMPLE ===")
print(json.dumps(samples[0], indent=2)[:2500])
