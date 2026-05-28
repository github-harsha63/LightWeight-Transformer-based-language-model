"""
my_tokenizer.py
---------------
Byte-level BPE tokenizer built from scratch.

Loads vocab.json and bpe_merges.txt which were produced during
the vocabulary-building phase (see main2.py).

Public API
----------
    tok = ByteTokenizer()
    ids  = tok.encode("Hello world")      # str  → list[int]
    text = tok.decode([15496, 995])       # list[int] → str
    t    = tok.to_tensor("Hello")         # str → LongTensor (T,)
"""

import os
import re
import json
from functools import lru_cache
from typing import List, Optional
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Byte ↔ unicode mapping
# Every possible byte value is mapped to a unique printable character so the
# BPE algorithm never has to deal with whitespace or control bytes mid-token.
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _byte_maps():
    nice = (
        list(range(ord('!'), ord('~') + 1)) +
        list(range(ord('¡'), ord('¬') + 1)) +
        list(range(ord('®'), ord('ÿ') + 1))
    )
    b2c, n = {}, 256
    for b in range(256):
        if b in nice:
            b2c[b] = chr(b)
        else:
            b2c[b] = chr(n); n += 1
    return b2c, {v: k for k, v in b2c.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Pre-tokenisation split pattern
# Splits raw text into word-like chunks before BPE merging begins.
# ─────────────────────────────────────────────────────────────────────────────

_SPLIT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?\w+"
    r"| ?\d+"
    r"| ?[^\s\w\d]+"
    r"|\s+(?!\S)"
    r"|\s+",
    re.UNICODE,
)


# ─────────────────────────────────────────────────────────────────────────────
# BPE merge logic
# ─────────────────────────────────────────────────────────────────────────────

def _pairs(seq: tuple) -> set:
    return {(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}


def _apply_merges(symbols: tuple, ranks: dict) -> tuple:
    while True:
        ps = _pairs(symbols)
        if not ps:
            break
        best = min(ps, key=lambda p: ranks.get(p, float('inf')))
        if best not in ranks:
            break
        a, b = best
        out, i = [], 0
        while i < len(symbols):
            try:
                j = symbols.index(a, i)
            except ValueError:
                out.extend(symbols[i:]); break
            out.extend(symbols[i:j])
            if j < len(symbols) - 1 and symbols[j + 1] == b:
                out.append(a + b); i = j + 2
            else:
                out.append(symbols[j]); i = j + 1
        symbols = tuple(out)
    return symbols


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer class
# ─────────────────────────────────────────────────────────────────────────────

class ByteTokenizer:
    """
    Byte-level BPE tokenizer.

    Parameters
    ----------
    vocab_file  : token-string → integer id   (vocab.json)
    merges_file : BPE merge pairs             (bpe_merges.txt)
    """

    def __init__(
        self,
        vocab_file  : str = 'vocab.json',
        merges_file : str = 'bpe_merges.txt',
    ):
        for f in (vocab_file, merges_file):
            if not os.path.exists(f):
                raise FileNotFoundError(
                    f"'{f}' not found. Run main2.py to build the vocabulary first."
                )

        with open(vocab_file, 'r', encoding='utf-8') as fh:
            self._tok2id: dict = json.load(fh)
        self._id2tok: dict = {v: k for k, v in self._tok2id.items()}

        self._ranks: dict = {}
        with open(merges_file, 'r', encoding='utf-8') as fh:
            for rank, line in enumerate(fh):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) == 2:
                    self._ranks[(parts[0], parts[1])] = rank

        self._b2c, self._c2b = _byte_maps()
        self._cache: dict    = {}
        self.vocab_size      = len(self._tok2id)

    # ── internals ─────────────────────────────────────────────────────────────

    def _word_to_ids(self, word: str) -> List[int]:
        bc   = ''.join(self._b2c[b] for b in word.encode('utf-8'))
        key  = tuple(bc)
        if key not in self._cache:
            self._cache[key] = _apply_merges(key, self._ranks)
        return [self._tok2id[t] for t in self._cache[key]]

    # ── public ────────────────────────────────────────────────────────────────

    def encode(self, text: str) -> List[int]:
        out = []
        for word in _SPLIT.findall(text):
            out.extend(self._word_to_ids(word))
        return out

    def decode(self, ids: List[int]) -> str:
        joined   = ''.join(self._id2tok[i] for i in ids)
        raw_bytes = bytes(self._c2b[ch] for ch in joined)
        return raw_bytes.decode('utf-8', errors='replace')

    def to_tensor(self, text: str, device: str = 'cpu') -> torch.Tensor:
        return torch.tensor(self.encode(text), dtype=torch.long, device=device)

    def batch_encode(
        self,
        texts     : List[str],
        device    : str = 'cpu',
        pad_len   : Optional[int] = None,
    ) -> torch.Tensor:
        id_lists = [self.encode(t) for t in texts]
        max_len  = pad_len or max(len(x) for x in id_lists)
        padded   = [x + [0] * (max_len - len(x)) for x in id_lists]
        return torch.tensor(padded, dtype=torch.long, device=device)


# ── self-test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    tok = ByteTokenizer()
    tests = [
        "Hello, world!",
        "The transformer architecture uses self-attention.",
        "Byte-pair encoding merges frequent character pairs.",
    ]
    print(f"Vocabulary size: {tok.vocab_size:,}\n")
    for s in tests:
        ids = tok.encode(s)
        rt  = tok.decode(ids)
        ok  = '✓' if rt == s else '✗'
        print(f"{ok}  [{len(ids):>3} tokens]  {s!r}")