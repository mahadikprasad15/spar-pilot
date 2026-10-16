"""Bounded batched CPU lens scoring; same normalized-list rules as Ticket03."""

import math
import numpy as np
from pilot_eval.trace_scoring import _spearman


class LensScorer:
    def __init__(
        self, norm, head, epsilon, inventory, references, matrix=None, batch_size=16
    ):
        self.norm, self.head, self.epsilon = norm, head, epsilon
        self.inventory, self.references, self.matrix = inventory, references, matrix
        self.ids = np.array(sorted(inventory.decoded), dtype=np.int64)
        self.batch_size = batch_size
        self.valid = [inventory.normalize(int(i)) for i in self.ids]
        groups = {}
        for index, word in enumerate(self.valid):
            if word is not None:
                groups.setdefault(word, []).append(index)
        self.support = {}
        for name, ref in references.items():
            rows = [
                r
                for r in ref["scores"]
                if r["y_a"] + r["y_b"] >= 5 and r["string"] in groups
            ]
            self.support[name] = (
                [(groups[r["string"]]) for r in rows],
                [r["z"] for r in rows],
            )

    def score_logits(self, logits, *, norm):
        if norm == 0:
            return dict(
                state="undefined",
                reason="zero-vector",
                norm=0,
                valid_count=0,
                scores={
                    name: {
                        k: None
                        for k in [
                            "overlap",
                            "overlap_word",
                            "overlap_format",
                            "soft_rank",
                        ]
                    }
                    for name in self.references
                },
            )
        if not np.isfinite(logits).all():
            raise ValueError("Nonfinite analysis vocabulary score.")
        count = min(1000, len(logits))
        # Select the same first1000 as a full lexsort, including boundary ties.
        threshold = np.partition(logits, len(logits) - count)[len(logits) - count]
        higher = np.flatnonzero(logits > threshold)
        equal = np.flatnonzero(logits == threshold)
        chosen = np.concatenate([higher, equal[: count - len(higher)]])
        chosen = chosen[np.lexsort((self.ids[chosen], -logits[chosen]))]
        strings = []
        token_ids = []
        seen = set()
        for index in chosen:
            word = self.valid[index]
            if word is not None and word not in seen:
                strings.append(word)
                token_ids.append(int(self.ids[index]))
                seen.add(word)
                if len(strings) == 20:
                    break
        scores = {}
        for name, ref in self.references.items():
            values = {
                field: len(seen & set(ref[key])) / len(strings) if strings else None
                for field, key in [
                    ("overlap", "strings"),
                    ("overlap_word", "word"),
                    ("overlap_format", "format_number"),
                ]
            }
            groups, z = self.support[name]
            soft = _spearman([float(np.max(logits[g])) for g in groups], z)
            values["soft_rank"] = soft if math.isfinite(soft) else None
            scores[name] = values
        return dict(
            state="scored" if strings else "undefined",
            reason=None if strings else "no-valid-strings",
            norm=float(norm),
            strings=strings,
            list_ids=token_ids,
            valid_count=len(strings),
            short_list=len(strings) < 20,
            scores=scores,
        )

    def many(self, vectors):
        vectors = np.asarray(vectors, dtype=np.float64)
        if (
            vectors.ndim != 2
            or vectors.shape[1] != len(self.norm)
            or not np.isfinite(vectors).all()
        ):
            raise ValueError(
                "Expected finite analysis vectors with pinned model width."
            )
        rows = []
        # Vocabulary columns are chunked; temporary score workspace is bounded by batch_size.
        for offset in range(0, len(vectors), self.batch_size):
            raw = vectors[offset : offset + self.batch_size]
            moved = raw if self.matrix is None else raw @ self.matrix.T
            lengths = np.linalg.norm(raw, axis=1)
            moved_zero = ~np.any(moved, axis=1)
            normalized = (
                moved
                * self.norm
                / np.sqrt(np.mean(moved * moved, axis=1) + self.epsilon)[:, None]
            )
            logits = np.empty((len(raw), len(self.ids)), dtype=np.float64)
            for start in range(0, len(self.ids), 4096):
                selected = self.ids[start : start + 4096]
                logits[:, start : start + len(selected)] = (
                    normalized @ np.asarray(self.head[selected], dtype=np.float64).T
                )
            for i, score in enumerate(logits):
                row = self.score_logits(score, norm=0 if moved_zero[i] else lengths[i])
                if moved_zero[i] and lengths[i] != 0:
                    row["reason"] = "zero-transport"
                rows.append(row)
        return rows
