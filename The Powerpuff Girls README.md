# TechJam 2026 Shopping Copilot

This is our submission for the TikTok TechJam 2026 conversational search track. It is a shopping agent that holds a multi-turn conversation with a customer, asks clarifying questions along the way, keeps track of what has been said and returns a ranked list of ten candidate products on every turn.

The whole thing is deterministic. It runs on SQLite FTS5 for retrieval, adds a term-coverage re-ranking layer of our own, accumulates session state across turns and decides what to ask next using logic we tuned empirically against the public set. There are no API calls anywhere in it, no model weights to load and nothing that needs a network connection at inference time.

---

## Results

Everything below comes from running `python3 -m evaluator.local_evaluator` against the 200-session public set in `data/public_set.jsonl`, and reproduces what is in `results.json`.

| Stage | HitRate@10 | MRR | MTTC | TechnicalScore |
|---|---|---|---|---|
| Organizer starter (weak BM25) | 0.125 | 0.068 | 9.81 | 0.107 |
| + session state, clarification, prioritization | 0.865 | 0.532 | 3.77 | 0.737 |
| + RRF term-coverage re-ranking | 0.890 | 0.543 | 3.51 | 0.758 |
| + term-tier weight tuning | **0.905** | **0.554** | **3.38** | **0.771** |

Broken down by scenario, the picture looks like this.

| Scenario | n | HitRate@10 | MRR | MTTC |
|---|---|---|---|---|
| Boundary | 10 | 0.900 | 0.642 | 3.40 |
| Browsing | 80 | 0.925 | 0.530 | 3.14 |
| Buying | 80 | 0.913 | 0.554 | 2.95 |
| Intent Override | 30 | 0.833 | 0.586 | 5.17 |

Intent Override is our weakest scenario, which is roughly what we would expect, since those sessions cannot register a hit until after the customer changes their mind on turn three or four.

---

## Setup

You will need Python 3.10 or newer. Everything else the agent uses comes from the standard library, so there is nothing to install with pip, and you will not need an API key or a network connection to run it.

Start by cloning the repository.

```bash
git clone <this-repo-url>
cd techjam-conversational-search
```

The catalog of 50,000 products is too large to commit, so it does not live in the repository. You can download it from the participant kit release, check it against the published hash, and move it into place with the following.

```bash
curl -L -O https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/catalog.jsonl.gz
curl -L -O https://github.com/TechJam2026/techjam-conversational-search/releases/download/participant-kit/SHA256SUMS
shasum -a 256 -c SHA256SUMS
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

If you are on Linux rather than macOS, use `sha256sum -c SHA256SUMS` for the check instead. Either way you should see a line confirming that `catalog.jsonl.gz` is OK.

---

## Reproducing our results

Run both of these from the repository root.

```bash
python3 -m evaluator.local_evaluator
```

This evaluates all 200 sessions in the public set and writes the final metrics to `results.json`. It should reproduce the numbers in the results table above exactly, because the agent is deterministic and repeated runs give identical output.

```bash
python3 benchmark_latency.py
```

This times `respond()` across every real turn of the public set and reports the mean, median, p95, and p99.

---

## How it works

### Retrieval

At startup we index all 50,000 catalog products into a SQLite FTS5 virtual table using the unicode61 tokenizer and query it through a field-weighted `bm25()` call. Titles carry the most weight at 6.0, followed by categories at 4.0, features and details at 2.5, store at 1.5 and description at 1.0. Alongside that we keep a plain-indexed table called `product_text`, keyed on `parent_asin` which lets the re-ranker pull per-field text for a small candidate set by primary key rather than scanning the FTS table's unindexed ID column.

### Session state

Rather than keeping one flat bag of terms, each session tracks three separate sets, and that separation turns out to matter more than we first expected.

The first is `category_terms`, which we fix on turn one and then never touch again. This is because it survives an intent override, the search cannot drift off into an unrelated category no matter what the customer says later on. The second is `learned_terms`, which holds everything disclosed so far and is what gives retrieval its breadth. The third is `disclosed_terms`, which holds only genuine constraint answers and deliberately excludes anything seeded from the customer profile.

Keeping these apart is what makes intent-override handling safe. When an override fires we clear `learned_terms` and `disclosed_terms` but leave `category_terms` alone, so the customer's change of mind rewrites their constraints without throwing away the category we established at the start.

### Re-ranking

A BM25 OR-query will surface a candidate on the strength of a single term match, which means it under-rewards the candidates that match more of the distinct query terms. To correct for that we compute a second ordering from a term-coverage score. That score weights terms by how much we trust them, giving disclosed constraint terms triple weight, turn-one category terms one and a half times, and generic learned or profile terms zero weight. That last one used to sit at a single weight until we went back and swept it — turns out any nonzero weight there let noisy profile-tag seeding (things like "comfort" or "fit" carried over from the customer's purchase history) dilute the score, and zeroing it out was worth 0.758 to 0.771 on its own. It also weights by which field the term appeared in, mirroring the BM25 field weights. We then fuse the two orderings with Reciprocal Rank Fusion, `1/(k+rank_bm25) + 1/(k+rank_coverage)`, where we settled on k=8 after a sweep.

Since RRF re-orders the candidate pool rather than filtering it, recall over that pool is preserved by construction. Hit rate can still shift if a target happens to sit right at the `top_k` boundary, so rather than assume we were safe we went and checked, and hit rate either held or improved at every configuration we tried.

### Clarifying questions

The agent walks a fixed attribute order that we tuned empirically, running through material, other, feature, color, style, use_case, size, budget and brand and skipping anything it has already asked about. It stops asking under three conditions. The first is when turns are running short and it has passed turn six. The second is when the last two replies were both uninformative and the next one looks unlikely to help. The third is simply when it has worked through every attribute. Whatever happens, it always sends a best-guess ranked list alongside the question, so it never goes quiet while waiting for the customer to answer.

---

## Model and cost

### There is no LLM in this pipeline

Our agent makes no API calls at any point, and on every turn it reports zero prompt tokens and zero completion tokens. That is not a placeholder we forgot to fill in. There is genuinely nothing to count.

This was a deliberate choice rather than a shortcut, and it bought us two things.

The first was cost. Every session runs at zero marginal cost, which meant we never had to think about rate limits, managing API keys, or tying ourselves to a particular vendor. During development it also meant we could re-run the full 200-session evaluation as often as we wanted, and that freedom is a large part of why we were able to iterate as quickly as we did.

The second matters more for scoring. The spec notes that organizers may disable network access during final evaluation. Any team relying on a hosted model needs a fallback path for that case, and a fallback is only ever as good as the testing behind it. Ours behaves identically whether or not there is a network connection, because nothing in it depends on one. There is no degraded mode for us to worry about, since there is no dependency that could degrade.

### Latency

| Metric | Value |
|---|---|
| Index build, one time at startup across 50k products | 1.64 s |
| `respond()` mean | 35.9 ms |
| `respond()` median | 32.9 ms |
| `respond()` p95 | 66.0 ms |
| `respond()` p99 | 72.2 ms |

A p99 of 72 ms means a full 200-session evaluation finishes in well under a minute and that is what let us re-run the harness after every single change rather than batching experiments up and testing them together.

### Tools, libraries and data

For development we used VS Code, Git and GitHub and the organizer's `local_evaluator.py` as our main measurement harness. We also wrote a few diagnostic scripts of our own for sizing candidate sets, timing latency and running the RRF parameter sweeps.

We used no external APIs at all. On the library side we stayed inside the Python standard library, running retrieval through the built-in `sqlite3` module and its FTS5 support, so there is no numpy, no scikit-learn, no PyTorch and no vector database anywhere in the project.

For data we used only what the organizers gave us, which is the frozen catalog of 50,000 products from the Amazon Reviews 2023 Clothing, Shoes and Jewelry category, together with the 200 labelled public development sessions in `public_set.jsonl`. We brought in no external training data and labelled nothing by hand.

---

## How we got here

We built and measured seven changes against the local harness this cycle. Five of them we reverted once we had numbers, and two survived. We have written up the reverts alongside the successes, because a list of things that worked tells you nothing about whether a result came from measurement or from luck.

### What shipped

Our RRF term-coverage re-ranking took the score from 0.737 to 0.758. We found the configuration through two sweeps, one over the RRF constant k and one over how large a candidate pool to feed the re-ranker. The best result came from k=8 with a pool five times the size of `top_k` and that was the local optimum across everything we tried.

The coverage score's tier weights had been hand-picked when we first built the re-ranker and never actually swept, so we went back and did that. Zeroing the generic learned-term tier came out on top by a clean margin, and it moved every metric at once: hit rate went from 0.890 to 0.905, MRR from 0.543 to 0.554, and MTTC dropped from 3.51 to 3.38, taking the overall score to 0.771. Zeroing that tier doesn't erase genuine disclosed constraints, since a term that shows up in both the learned and disclosed sets still gets the disclosed weight — the max across tiers wins. It only strips out terms that are exclusively generic profile-tag seeding. Intent Override sessions were untouched either way, since those already clear the learned-term set the moment an override fires.

### What we reverted and why

**Intent-weighted retrieval fusion**

- We tried shifting the RRF blend toward BM25 for buying sessions and toward coverage for browsing sessions, borrowing the shape of a BM25 and dense-embedding hybrid. It regressed to 0.745.
- Looking at why, the problem is that we had no genuine second signal to condition on. Our coverage score is already the precise, disclosed-constraint-driven signal that BM25 plays the role of in a real hybrid, so tuning away from an even balance could only cost us.

**Adaptive pool-stagnation stopping**

- The thought here was to stop asking clarifying questions once the candidate pool had failed to shrink across two consecutive asks. It regressed to 0.720.
- Our existing stop conditions, meaning the no-information reply streak and the turn cap, were already capturing that signal, and they did it without the false positives that comparing pool sizes turn over turn introduces on sessions that are naturally flat. Boundary hit rate fell from 0.900 to 0.700 under the change, which is what gave it away.

**Information-gain clarifying-question selection**

- Instead of a fixed order, we tried picking whichever attribute would most split the live candidate pool. Our first version collapsed to somewhere between 0.34 and 0.54, which turned out to be a genuine bug rather than a bad idea. It was starving the `other` attribute, which the evaluator's simulated customer treats as a wildcard that always yields an answer.
- Once fixed, it still topped out at 0.750 and sat below baseline at every threshold we tested. The fixed order we already had was tuned against this customer's answering patterns, and pool diversity turns out to be a different and less useful signal than whether the customer will answer at all.

**Boilerplate noise filtering**

- We tried skipping tokenization on template replies like "I don't have an additional preference for X", so that filler words such as *additional* and *specific* would stop polluting the query. It looked like an obvious bug fix. It regressed to 0.737.
- It recovered none of the 22 misses we had aimed it at, because those same words show up in ordinary Amazon listing copy, in phrases like "Special Feature" and "Additional details", and were quietly carrying real signal for several borderline sessions sitting between rank 8 and rank 10.

**Price and budget as a second retrieval route**

- We looked into using the catalog's numeric price field as a non-lexical signal routed alongside BM25. It was dead on arrival for this dataset.
- The local evaluator's simulated customer does construct a budget constraint for 178 of the 200 targets, but it always appends that constraint after several feature and detail items in the intent card, so it never survives the `hard_constraints[:2]` and `soft_preferences[2:4]` slice. Zero of the 178 made it through, which means no algorithm built on that signal could ever have fired against this harness.

---

## Where it falls short

**Near-duplicate products are where our ranking breaks down**

- We watched it happen on a leather belt, which lands 226th out of 258 even after the category filter has already narrowed things to other belts. Its listing repeats "leather" and "buckle" throughout, and so do the 225 or so competitors sitting above it, so term-frequency scoring has nothing left to separate them with.
- Getting past this would take a real semantic signal rather than smarter keyword weighting, and our stack has no way to produce one. We chose to stay dependency-free and run without numpy or sklearn, and this is the cost of that choice.

**We are pattern-matching the customer's language**

- Both `_is_override` and `no_info_reply` key off phrasings we saw in the evaluator.
- We think they generalise reasonably well, since the phrases involved are ordinary ones like "ignore my earlier preference", but we have never seen the language the private 800-session simulator uses and cannot honestly claim more than that.

**A few of our findings might only hold locally**

- The price-route dead end came out of how `intent_card()` builds cards from `public_set.jsonl`, and the private set may well ship organizer-provided cards that disclose things differently. If that is the case, the finding may not transfer, and private-set access would be the only way to know for sure.
- The same caveat applies to our fixed attribute order, which is tuned against a simulator we can see.

**We are at 0.905, short of the 0.95 we set out to reach**

- Six attempts this session went at the ceiling, each aimed at a cause we had actually diagnosed rather than a hunch. Five of them traded away sessions that were already working and got reverted; one, retuning the coverage score's tier weights, moved the ceiling from 0.89 to 0.905.
- That most of them still failed points to a structural limit rather than a shortage of ideas, though we would want more time before saying so with real confidence.

### What we would do with more time

**Add a semantic signal for near-duplicates**

- The near-duplicate ceiling is the obvious first thing to go after, and getting past it needs a semantic signal rather than better keyword weighting.
- A small sentence-embedding model over titles and features, used to re-rank the same candidate pool we already build, would slot into our architecture without disturbing anything upstream of it.

**Learn the question policy instead of fixing it**

- We would replace the fixed attribute order with a policy learned from real customer responses rather than one tuned against a single known simulator.
- That is the version of our information-gain experiment that would actually have something worth learning from.

**Validate the language patterns more widely**

- We would go and validate the override and no-information patterns against a much wider range of phrasings than the local evaluator produces.
- That is easily our least defensible assumption at the moment, and it is the one most likely to cost us on the private set.

---

## Team member contributions

Ashley ran local evaluation and benchmarking, designed and compared the experiments, analysed the reverted approaches and failure cases, and documented results and performance.

Janson handled system integration and testing, reproducibility and final validation, the technical report and demo preparation, and the presentation materials.

Jia Xin designed the session state and term tracking, built intent-override handling, wrote the clarifying-question logic and its stopping conditions and handled no-information responses.

Ivy built the retrieval pipeline and the SQLite FTS5 and BM25 implementation, wrote the term-coverage scoring and RRF re-ranking, tuned the RRF parameters and candidate pool and analysed ranking failures.
