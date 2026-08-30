import json
import sys
sys.path.insert(0, ".")
from starter.agent import Agent, _is_override, _terms
from evaluator.local_evaluator import (
    load_jsonl, catalog_index, coarse_category, initial_message,
    customer_reply, materialize_hidden_fields, normalize_recommendations,
    MAX_TURNS, TOP_K,
)

samples = load_jsonl("data/public_set.jsonl")
catalog_ids, categories, products = catalog_index("data/catalog.jsonl")

# pick a couple of failing intent_override sample_ids from your results.json
target_ids = {"public_0002", "public_0003", "public_0013"}

agent = Agent("data/catalog.jsonl")

for sample in samples:
    if sample["sample_id"] not in target_ids:
        continue
    print("=" * 70)
    print("sample_id:", sample["sample_id"])
    session_id = f"debug_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": card, "behavior": behavior}
    disclosed = set()
    boundary_used = False
    override_applied = False
    user_message = initial_message(effective_sample, coarse_category(categories.get(target, [])), disclosed)
    print("target_asin:", target)
    print("override info:", behavior.get("override"))
    for turn in range(1, MAX_TURNS + 1):
        print(f"-- turn {turn} --")
        print("  user_message:", user_message)
        response = agent.respond(session_id, user_message, turn, TOP_K)
        print("  ask_attribute:", response.get("ask_attribute"))
        ranked = normalize_recommendations(response.get("recommendations"), catalog_ids)
        print("  target in top10:", target in ranked, "| rank:", (ranked.index(target)+1) if target in ranked else None)
        if override_applied and target in ranked:
            print("  HIT")
            break
        if turn == MAX_TURNS:
            break
        override = behavior.get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            user_message = str(override.get("message"))
        else:
            user_message, boundary_used = customer_reply(effective_sample, response.get("ask_attribute"), disclosed, boundary_used)
    print()
