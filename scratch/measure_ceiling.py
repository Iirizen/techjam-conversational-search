"""Measure how far cheap deterministic retrieval gets you.

Run from the repo root (needs the evaluator package importable):
    python3 scratch/measure_ceiling.py

Reports:
  A. Category ceiling  - candidates left after filtering on the turn-1 category.
  B. Parse round-trip   - can you reliably recover that category from the message?
  C. Constraint power   - how much a single disclosed constraint narrows things.
"""

from __future__ import annotations

import re
import statistics
from collections import defaultdict

from evaluator.local_evaluator import (
    catalog_index,
    coarse_category,
    load_jsonl,
    initial_message,
    materialize_hidden_fields,
    searchable_text,
)

CATALOG = "data/catalog.jsonl"
DATASET = "data/public_set.jsonl"
GLOBAL_SAMPLE = 50  # sessions used for the expensive whole-catalog scan


def percentiles(values: list[int]) -> str:
    ordered = sorted(values)
    if not ordered:
        return "n/a"

    def at(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return (
        f"min {ordered[0]}  p25 {at(.25)}  median {at(.50)}  "
        f"p75 {at(.75)}  p90 {at(.90)}  max {ordered[-1]}"
    )


print("loading catalog...")
catalog_ids, categories, products = catalog_index(CATALOG)
samples = load_jsonl(DATASET)
print(f"  {len(catalog_ids)} products, {len(samples)} sessions\n")

# Reverse index: coarse category -> asins
by_category: dict[str, list[str]] = defaultdict(list)
for asin, cats in categories.items():
    by_category[coarse_category(cats)].append(asin)

print(f"distinct coarse categories: {len(by_category)}")
bucket_sizes = sorted((len(v) for v in by_category.values()), reverse=True)
print(f"  largest buckets: {bucket_sizes[:8]}")
print(f"  singleton buckets: {sum(1 for n in bucket_sizes if n == 1)}\n")

# ------------------------------------------------------ A. category ceiling
print("=== A. CATEGORY CEILING ===")
candidate_counts: list[int] = []
for sample in samples:
    target = str(sample["ground_truth"]["parent_asin"])
    category = coarse_category(categories.get(target, []))
    candidate_counts.append(len(by_category[category]))

print("candidates surviving the turn-1 category filter:")
print("  " + percentiles(candidate_counts))
median = statistics.median(candidate_counts)
print(f"\n  median = {median:.0f}")
if median < 200:
    print("  -> VERDICT: deterministic retrieval is strong. Index offline, use an")
    print("     LLM only to re-rank a short list.")
elif median < 2000:
    print("  -> VERDICT: category + constraint matching, with embeddings over the")
    print("     filtered set.")
else:
    print("  -> VERDICT: category alone is weak. Dense retrieval must be the backbone.")

# ---------------------------------------------------- B. parse round-trip
print("\n=== B. PARSE ROUND-TRIP ===")
OPENING = re.compile(r"^I'm looking for (.+?)(?:, but I'm still exploring\.|\. )")

ok = bad = 0
failures: list[tuple[str, str]] = []
for sample in samples:
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}
    truth = coarse_category(categories.get(target, []))
    message = initial_message(effective, truth, set())
    match = OPENING.match(message)
    if match and match.group(1) == truth:
        ok += 1
    else:
        bad += 1
        if len(failures) < 5:
            failures.append((truth, message[:120]))

print(f"  recovered correctly: {ok}/{len(samples)}  ({ok / len(samples):.1%})")
if failures:
    print("  sample failures (true category | message):")
    for truth, message in failures:
        print(f"    {truth!r} | {message!r}")
    print("  -> widen the regex before relying on it")

# ------------------------------------------------- C. constraint power
print("\n=== C. CONSTRAINT POWER ===")
print("  (how many products contain the first disclosed constraint verbatim)")

lowered_text = {asin: searchable_text(p).lower() for asin, p in products.items()}

within_counts: list[int] = []
global_counts: list[int] = []
unmatched = 0

for index, sample in enumerate(samples):
    target = str(sample["ground_truth"]["parent_asin"])
    card, _ = materialize_hidden_fields(sample, products)
    hard = card.get("hard_constraints") or []
    if not hard:
        continue
    needle = str(hard[0]).lower()
    category = coarse_category(categories.get(target, []))

    hits_in_category = sum(1 for a in by_category[category] if needle in lowered_text[a])
    within_counts.append(hits_in_category)

    if needle not in lowered_text[target]:
        unmatched += 1  # constraint came from a dict rendering; substring fails

    if index < GLOBAL_SAMPLE:
        global_counts.append(sum(1 for t in lowered_text.values() if needle in t))

print("\n  within the target's category:")
print("    " + percentiles(within_counts))
print(f"\n  across the whole catalog (first {len(global_counts)} sessions):")
print("    " + percentiles(global_counts))

total = len(within_counts)
print(f"\n  constraints NOT found verbatim in their own target: {unmatched}/{total}"
      f"  ({unmatched / total:.1%})" if total else "")
print("  -> that percentage is the share of constraints where substring matching")
print("     fails outright (dicts render as 'key: value' in the card but")
print("     'key value' in searchable_text). Those need fuzzy matching.")
