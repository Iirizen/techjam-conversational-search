import json
from collections import Counter

with open("results.json") as f:
    data = json.load(f)

by_scenario = {}
for s in data["sessions"]:
    scenario = s["scenario_type"]
    by_scenario.setdefault(scenario, []).append(s)

print("Rank distribution among HITS (where in top-10 the target landed):")
print()
for scenario, sessions in sorted(by_scenario.items()):
    hits = [s for s in sessions if s["hit"]]
    misses = len(sessions) - len(hits)
    rank_counts = Counter(s["best_rank"] for s in hits)
    print(f"-- {scenario} -- ({len(hits)} hits, {misses} misses out of {len(sessions)})")
    for rank in range(1, 11):
        count = rank_counts.get(rank, 0)
        if count:
            bar = "#" * count
            print(f"  rank {rank:2d}: {count:3d}  {bar}")
    print()

# Overall: how many hits are rank 1 vs rank 6-10 (the ones dragging MRR down)
all_hits = [s for s in data["sessions"] if s["hit"]]
rank1 = sum(1 for s in all_hits if s["best_rank"] == 1)
rank_2_5 = sum(1 for s in all_hits if 2 <= s["best_rank"] <= 5)
rank_6_10 = sum(1 for s in all_hits if 6 <= s["best_rank"] <= 10)
print(f"Overall hits: {len(all_hits)}")
print(f"  rank 1:     {rank1} ({rank1/len(all_hits)*100:.1f}%)")
print(f"  rank 2-5:   {rank_2_5} ({rank_2_5/len(all_hits)*100:.1f}%)")
print(f"  rank 6-10:  {rank_6_10} ({rank_6_10/len(all_hits)*100:.1f}%)")
