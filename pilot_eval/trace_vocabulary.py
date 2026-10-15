"""Outcome-blind token normalization and task-reference construction."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


class TokenInventory:
    """Explicit ID-to-decoded-string inventory; no model or network access."""

    def __init__(self, decoded, *, added_ids=(), special_ids=()):
        self.decoded = dict(decoded)
        self.excluded = set(added_ids) | set(special_ids)
        self.normalized = {}
        for token_id, text in self.decoded.items():
            value = unicodedata.normalize('NFKC', text).strip().lower()
            self.normalized[token_id] = (value if token_id not in self.excluded
                and value and '\ufffd' not in value else None)

    def normalize(self, token_id):
        if token_id not in self.normalized:
            raise ValueError(f'Token ID {token_id} is absent from the pinned inventory.')
        return self.normalized[token_id]

    def count(self, token_ids):
        return dict(Counter(value for token in token_ids
                            if (value := self.normalize(token)) is not None))
