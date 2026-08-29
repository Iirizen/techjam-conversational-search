# TechJam Conversational E-Commerce Search — Report

## 1. Summary

The agent (`starter/agent.py`) is a fully deterministic, LLM-free conversational
shopping assistant built on SQLite FTS5 (BM25) with a custom term-coverage
re-ranking layer, session-state accumulation, and empirically-tuned
clarifying-question logic. No API calls, no model weights, no network
dependency at inference time.

See `DEMO.md` for a full turn-by-turn transcript of a real session
(intent-override handling + non-repeating clarifying questions), generated
by `generate_demo_transcript.py`.

| Stage | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Organizer starter (weak BM25) | 0.125 | 0.068 | 9.81 | ~0.107 |
| + session state, clarification, prioritization | 0.865 | 0.532 | 3.77 | 0.737 |
| + RRF term-coverage re-ranking | 0.890 | 0.543 | 3.51 | 0.758 |
| + term-tier tuning (final) | **0.905** | **0.554** | **3.38** | **0.771** |

Scores are from `python3 -m evaluator.local_evaluator` on the 200-session
public set (`data/public_set.jsonl`), reproducing `results.json`.

Scenario breakdown (final):

| Scenario | n | HitRate@10 | MRR | MTTC |
|---|---|---|---|---|
| Boundary | 10 | 0.900 | 0.642 | 3.40 |
| Browsing | 80 | 0.925 | 0.530 | 3.14 |
| Buying | 80 | 0.913 | 0.554 | 2.95 |
| Intent Override | 30 | 0.833 | 0.586 | 5.17 |

Intent Override is unchanged by the latest tuning pass — those sessions
already discard `learned_terms` on override, so the change that helped
elsewhere doesn't touch them either way.

## 2. Architecture

**Retrieval.** All 50,000 catalog products are indexed once at startup into a
SQLite FTS5 virtual table (`unicode61` tokenizer), with a field-weighted
`bm25()` query (title 6.0, categories 4.0, features/details 2.5, store 1.5,
description 1.0). A companion plain-indexed table (`product_text`, keyed on
`parent_asin`) lets the re-ranker fetch per-field text for a small candidate
set by primary key, instead of scanning the FTS table's unindexed ID column.

**Session state.** Each session tracks three term sets, not one flat bag:
`category_terms` (fixed from turn 1, survives intent overrides so the search
never drifts into an unrelated category), `learned_terms` (everything
disclosed so far, used for retrieval breadth), and `disclosed_terms` (real
constraint answers only — excludes profile-tag seeding). This separation is
what makes intent-override handling safe: on an override, only
`learned_terms`/`disclosed_terms` reset; category context is preserved.

**Re-ranking (the main technical contribution this cycle).** The initial
BM25 OR-query only needs one term match to surface a candidate, which
under-rewards candidates that match *more* of the distinct query terms. We
compute a second ordering — term-coverage score, weighted by term
reliability (disclosed constraint terms 3×, turn-1 category terms 1.5×,
generic learned/profile terms **0×**, tuned by sweep — see §4) and by which
field the term appears in (mirroring the BM25 field weights) — then fuse
the two orderings with **Reciprocal Rank Fusion**
(`1/(k+rank_bm25) + 1/(k+rank_coverage)`, k=8, tuned by sweep). Because RRF
re-orders the existing candidate pool rather than filtering it, hit-rate
can only hold or improve, never regress from this stage — confirmed
empirically (see §4).

Zeroing the generic-term tier doesn't erase genuinely disclosed
constraints: `_term_tiers` takes a `max()` across tiers, so a term that
appears in both `learned_terms` and `disclosed_terms` still gets the 3×
disclosed weight. It only zeroes out terms that are *exclusively* in
`learned_terms` — i.e. the profile-tag seeding from `reset()` (generic
tags like "comfort"/"fit" that describe the customer's history, not
necessarily this specific target).

**Clarifying questions.** A fixed, empirically-tuned attribute order
(`material, other, feature, color, style, use_case, size, budget, brand`)
is walked once per turn, skipping attributes already asked. The agent stops
asking once turns are running out (`turn > 6`), the next answer is unlikely
to be informative (two consecutive non-informative replies), or every
attribute has been asked. A best-guess ranked list is always returned
alongside the question — the agent never goes silent while waiting on the
customer.

## 3. Model & Cost Disclosure

**No LLM is used anywhere in this pipeline.** `usage` is reported as
`{"prompt_tokens": 0, "completion_tokens": 0}` on every turn, and this is
literally true, not a rounding artifact — there are no API calls to make.

This has two consequences worth stating explicitly for judging:

- **Feasibility**: zero marginal cost per session, no rate limits, no
  vendor lock-in, no API key management.
- **Robustness to the stated official-scoring risk**: the spec notes
  organizer policy may disable network access for final scoring. This
  agent's behavior is identical online or fully air-gapped, by
  construction — nothing to fall back to, because nothing depends on it.

**Latency** (measured locally with `benchmark_latency.py`, run against
every real turn of the 200-session public set — see repo for the script):

| Metric | Value |
|---|---|
| Index build (one-time startup, 50k products) | 1.69 s |
| `respond()` mean | 37.0 ms |
| `respond()` median | 33.0 ms |
| `respond()` p95 | 67.0 ms |
| `respond()` p99 | 82.9 ms |

These are indicative (this machine, not the organizer's eval environment),
but the shape of the claim is what matters: there is no network round-trip
to add on top, because there is no network call in the loop at all.

## 4. Engineering Process

Nine changes were built and evaluated against the local harness this
cycle; two shipped, seven were reverted after measurement. Recording the
negative results here deliberately — they're evidence of the decision
process, not just the outcome.

**Shipped:**
- *RRF term-coverage re-ranking* (§2) — 0.737 → 0.758, tuned via a k-sweep
  (k=4..30) and pool-size sweep; k=8 with a 5×top_k candidate pool was the
  local optimum.
- *Term-tier weight tuning* — 0.758 → 0.771. The coverage score's tier
  weights (category/learned/disclosed) had been hand-picked when the
  re-ranker was first built and never actually swept. A joint sweep found
  zeroing the generic `learned_terms` tier (down from 1×) was a clean win
  on every metric simultaneously — hit-rate, MRR, and MTTC all improved,
  with Intent Override sessions unaffected either way (see §2 for why).

**Tried, reverted (with measured cause):**
- *Intent-weighted retrieval fusion* — shifting the RRF blend ratio toward
  BM25 for "buying" sessions and toward coverage for "browsing" sessions
  (mirroring a bm25/dense-embedding hybrid design). Regressed to 0.745 at
  best. Root cause: there's no real second signal here to condition on —
  the "coverage" score is already the precise, disclosed-constraint-driven
  signal (the BM25-analogue in a true hybrid), so detuning its weight away
  from the already-optimal 1:1 ratio only hurt.
- *Adaptive pool-stagnation stopping* — stop asking clarifying questions if
  the candidate pool hasn't shrunk across two asks. Regressed to 0.720.
  The existing stop conditions (no-info-reply streak, turn cap) already
  capture this signal without the false positives that turn-over-turn pool
  comparison introduces on naturally-flat sessions (e.g. boundary hit-rate
  dropped 0.9→0.7).
- *Information-gain clarifying-question selection* — pick whichever
  attribute would most split the live candidate pool instead of a fixed
  order. First version collapsed to 0.34–0.54 (a real bug: it starved the
  `"other"` attribute, which the evaluator's simulated customer treats as a
  wildcard that always yields an answer). Fixed version still topped out
  at 0.750 — below baseline at every threshold tested. The fixed attribute
  order isn't arbitrary; it's already tuned against this exact simulated
  customer's answer patterns, and pool-diversity is a different, less
  relevant signal than "will this synthetic customer actually answer."
- *Boilerplate noise filtering* — skip tokenizing template replies like
  "I don't have an additional preference for X" so filler words like
  `additional`/`specific` stop polluting the query. Looked like an obvious
  bug fix; regressed to 0.737 and recovered zero of the 22 misses it
  targeted, because those words also occur as ordinary Amazon listing
  copy ("Special Feature:", "Additional details:") and were providing real
  signal for several borderline (rank 8–10) sessions.
- *Price/budget as a second retrieval route* — investigated using the
  catalog's numeric `price` field as a non-lexical signal, hybrid-routed
  alongside BM25. Verified dead on arrival for this dataset: the local
  evaluator's synthetic customer constructs a budget constraint for 178/200
  targets, but it's always appended after several feature/detail items in
  the intent card and never survives the `hard_constraints[:2]` /
  `soft_preferences[2:4]` slice — 0/178, empirically confirmed. No
  algorithm built on top of it could ever fire against this harness.
- *IDF-weighted coverage scoring (blanket)* — used SQLite's `fts5vocab`
  table to weight each matched term by real catalog-wide inverse document
  frequency, on the theory that rare/distinctive terms should count for
  more than common ones (motivated directly by the near-duplicate ranking
  ceiling in §5). Regressed monotonically with weighting strength, down to
  0.717 at full strength. Root cause: this catalog is a single narrow
  domain (clothing/shoes/jewelry only), so the words that correctly anchor
  category identity — "belt", "leather", "shirt" — are *necessarily*
  common within it. Global IDF penalized exactly the category-anchoring
  signal the tier weights exist to protect.
- *IDF-weighted coverage scoring (scoped to disclosed terms)* — same idea,
  scoped to skip `category_terms` so category anchoring couldn't be
  diluted. Better than the blanket version but still never beat baseline
  (flat-to-negative at every exponent tested, 0.757 best case). Even
  customer-disclosed constraint terms (material/color names) are drawn
  from a small, common, everyday descriptive vocabulary in this domain —
  there's no real "rare but relevant term" phenomenon here for IDF to
  exploit.

## 5. Limitations

- **BM25 ranking ceiling on near-duplicate products.** Diagnosed directly:
  a leather belt target ranks 226th of 258 *even inside a category-filtered
  pool* of other belts, despite its listing repeating "leather" and
  "buckle" heavily — term-frequency scoring can't discriminate it from
  ~225 similarly-worded competitors. Closing this gap needs a genuine
  semantic/embedding signal, which this stack doesn't have (no numpy/
  sklearn; kept dependency-free deliberately).
- **Heuristic override/no-info detection** (`_is_override`, `no_info_reply`)
  is pattern-matched against the evaluator's known phrasings. It's
  reasonably generalizable (common phrases like "ignore my earlier
  preference"), but hasn't been validated against the private 800-session
  set's actual customer-simulator language.
- **Some diagnostic findings are local-evaluator-specific.** The
  price-route dead-end (§4) is a property of `public_set.jsonl`'s synthetic
  `intent_card()` construction, not necessarily of the private evaluation
  set, which may use organizer-provided intent cards with different
  disclosure behavior. Not re-testable without private-set access.
- **Current hit-rate ceiling is 0.905**, not the 0.95 initially targeted.
  Of nine independent, root-cause-diagnosed attempts to push past it this
  cycle, one (term-tier tuning, §4) moved the ceiling; the other seven
  regressed other sessions and were reverted. The pattern across all seven
  failures points to a structural limit of keyword/BM25 matching on
  near-duplicate products (§5) rather than a shortage of tuning ideas,
  though more time would be needed to say that with full confidence.

## 6. Reproduction

```bash
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl  # verify against SHA256SUMS
python3 -m evaluator.local_evaluator   # writes results.json
python3 benchmark_latency.py           # latency numbers in §3
```

Python 3.10+, standard library only — no `requirements.txt` needed for the
agent itself.

## 7. Team Contributions

- **Ashley** — local evaluation and benchmarking; designed and compared
  experiments; analyzed reverted approaches and failure cases; results and
  performance documentation.
- **Janson** — system integration and testing; reproducibility and final
  validation; technical report and demo preparation; presentation
  materials.
- **Jia Xin** — session state and term tracking design; intent-override
  handling; clarifying-question logic and stopping conditions; no-info
  response handling.
- **Ivy** — retrieval pipeline (SQLite FTS5/BM25); term-coverage scoring
  and RRF re-ranking; RRF and candidate-pool parameter tuning; ranking
  failure analysis.
