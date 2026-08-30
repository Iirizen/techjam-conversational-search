import json
from collections import Counter

counts = Counter()
with open('data/public_set.jsonl') as f:
    for line in f:
        line = line.strip()
        if line:
            s = json.loads(line)
            counts[s.get('scenario_type')] += 1

for k, v in counts.items():
    print(repr(k), ':', v)
