from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
    "still", "exploring", "actually", "ignore", "earlier", "preference",
    "what", "need", "key", "requirement", "prioritize", "target", "requirements",
}

OVERRIDE_PATTERNS = (
    "ignore my earlier",
    "actually, ignore",
    "what i need is",
)

ATTRIBUTE_ORDER = [
    "material", "other", "feature", "color", "style", "use_case", "size", "budget", "brand",
]


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


def _is_override(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in OVERRIDE_PATTERNS)


class _SessionState:
    __slots__ = ("category_terms", "learned_terms", "asked_attributes", "turn_count", "seen_first_turn", "no_info_streak")

    def __init__(self) -> None:
        self.category_terms: list[str] = []
        self.learned_terms: list[str] = []
        self.asked_attributes: set[str] = set()
        self.turn_count: int = 0
        self.seen_first_turn: bool = False
        self.no_info_streak: int = 0

    def all_terms(self) -> list[str]:
        combined = list(self.category_terms)
        for term in self.learned_terms:
            if term not in combined:
                combined.append(term)
        return combined

    def add_learned(self, new_terms: list[str]) -> None:
        for term in new_terms:
            if term not in self.learned_terms:
                self.learned_terms.append(term)

    def override_learned(self, new_terms: list[str]) -> None:
        # Drop stale constraint answers but keep category context intact.
        self.learned_terms = []
        self.add_learned(new_terms)


class Agent:
    """Stateful BM25 agent: accumulates constraints across turns, preserves
    category context through intent overrides, and asks non-repeating
    clarifying questions when the candidate pool is too broad."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: dict[str, _SessionState] = {}
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        state = _SessionState()
        if isinstance(user_profile, dict):
            tags = user_profile.get("preference_tags")
            if isinstance(tags, list):
                state.add_learned(_terms(" ".join(str(tag) for tag in tags)))
        self._sessions[session_id] = state

    def _search(self, terms: list[str], limit: int) -> list[str]:
        unique_terms = list(dict.fromkeys(terms))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            return []
        rows = self.connection.execute(
            "SELECT parent_asin FROM products WHERE products MATCH ? "
            "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
            (expression, limit),
        ).fetchall()
        return [str(row[0]) for row in rows]

    def _next_attribute(self, state: _SessionState) -> str | None:
        for attribute in ATTRIBUTE_ORDER:
            if attribute not in state.asked_attributes:
                return attribute
        return None

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        state.turn_count = turn

        new_terms = _terms(user_message)
        no_info_reply = (
            "i don't have" in user_message.lower()
            or "not quite right" in user_message.lower()
        )
        if no_info_reply and state.seen_first_turn:
            state.no_info_streak += 1
        elif state.seen_first_turn:
            state.no_info_streak = 0

        if not state.seen_first_turn:
            # Turn 1 establishes category context -- always kept, even
            # across an intent override later in the session.
            state.category_terms = list(dict.fromkeys(new_terms))
            state.seen_first_turn = True
        elif _is_override(user_message):
            # Customer explicitly discarded earlier preferences: drop only
            # the learned constraint answers, keep category context so the
            # search doesn't drift into an unrelated category.
            state.override_learned(new_terms)
            state.asked_attributes.clear()
        else:
            state.add_learned(new_terms)

        search_terms = state.all_terms()
        wide_pool = self._search(search_terms, limit=max(top_k * 5, 50))
        recommendations = [{"parent_asin": asin} for asin in wide_pool[:top_k]]

        ask_attribute = None
        message = "Here are the closest matches I found."

        candidate_pool_size = len(wide_pool)
        turns_left = 10 - turn
        next_attr = self._next_attribute(state)
        should_ask = (
            turns_left > 0
            and next_attr is not None
            and (candidate_pool_size == 0 or candidate_pool_size > top_k * 2)
            and turn <= 6
            and state.no_info_streak < 2
        )

        if should_ask:
            ask_attribute = next_attr
            state.asked_attributes.add(ask_attribute)
            message = f"Do you have a preference for {ask_attribute.replace('_', ' ')}?"

        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
