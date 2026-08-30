import statistics
import sys
import time
import uuid

sys.path.insert(0, ".")  # run from the repo root; scripts/ isn't on sys.path by default

from evaluator import local_evaluator as ev
from starter.agent import Agent

samples = ev.load_jsonl("data/public_set.jsonl")
catalog_ids, categories, products = ev.catalog_index("data/catalog.jsonl")

# --- One-time startup cost: building the FTS5 index over the full catalog ---
start = time.perf_counter()
agent = Agent("data/catalog.jsonl")
startup_s = time.perf_counter() - start

# --- Steady-state per-turn latency across every real session/turn ---
respond_latencies_ms: list[float] = []
first_turn_latencies_ms: list[float] = []
later_turn_latencies_ms: list[float] = []

for sample in samples:
    session_id = f"bench_{uuid.uuid4().hex}"
    agent.reset(session_id, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    effective_intent_card, effective_behavior = ev.materialize_hidden_fields(sample, products)
    effective_sample = {**sample, "intent_card": effective_intent_card, "behavior": effective_behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = ev.initial_message(effective_sample, ev.coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, ev.MAX_TURNS + 1):
        t0 = time.perf_counter()
        response = agent.respond(session_id, user_message, turn, ev.TOP_K)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        respond_latencies_ms.append(elapsed_ms)
        (first_turn_latencies_ms if turn == 1 else later_turn_latencies_ms).append(elapsed_ms)

        ranked = ev.normalize_recommendations(response.get("recommendations"), catalog_ids)
        if override_applied and target in ranked:
            break
        if turn == ev.MAX_TURNS:
            break
        override = effective_sample.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = ev.customer_reply(
                effective_sample, response.get("ask_attribute"), disclosed, boundary_used
            )


def pct(data: list[float], p: float) -> float:
    data = sorted(data)
    k = (len(data) - 1) * p
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    return data[f] + (data[c] - data[f]) * (k - f)


print(f"Catalog size: 50,000 products")
print(f"Startup (index build, one-time): {startup_s:.3f} s")
print()
print(f"respond() calls measured: {len(respond_latencies_ms)}")
print(f"  mean:   {statistics.mean(respond_latencies_ms):.2f} ms")
print(f"  median: {statistics.median(respond_latencies_ms):.2f} ms")
print(f"  p95:    {pct(respond_latencies_ms, 0.95):.2f} ms")
print(f"  p99:    {pct(respond_latencies_ms, 0.99):.2f} ms")
print(f"  max:    {max(respond_latencies_ms):.2f} ms")
print()
print(f"turn-1 calls (n={len(first_turn_latencies_ms)}): mean {statistics.mean(first_turn_latencies_ms):.2f} ms")
print(f"turn 2+ calls (n={len(later_turn_latencies_ms)}): mean {statistics.mean(later_turn_latencies_ms):.2f} ms")
