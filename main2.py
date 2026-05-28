"""
main2.py  –  Vocabulary construction + Pre-training
-----------------------------------------------------
Phase 1 : Build a byte-level BPE vocabulary from a large web-text corpus.
          Saves  vocab.json  and  bpe_merges.txt  to disk.

Phase 2 : Initialise a transformer language model from random weights,
          train it on the same corpus with a cosine-decayed AdamW schedule,
          and save the final checkpoint to  model_weights.pt.

Dataset  : OpenWebText  (38 GB of text scraped from URLs shared on Reddit)
Tokenizer: Byte-level BPE,  vocab_size = 50257,  built from the corpus
Model    : 12-layer transformer,  768-dim,  12 heads  ≈ 124M parameters
Compute  : ~300 000 gradient steps on 8× A100 GPUs

NOTE: The checkpoint produced by this script is already stored in
      model_weights.pt.  There is no need to re-run this file.
"""

import os
import re
import json
import math
import time
import random
import collections
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import IterableDataset, DataLoader


# PHASE 1  –  BPE vocabulary builder
TARGET_VOCAB_SIZE = 50257          # target number of BPE tokens
CORPUS_SAMPLE     = 500_000        # documents sampled for vocab construction

_PRESPLIT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+",
    re.UNICODE,
)

# Byte → printable-unicode mapping (avoids whitespace/control bytes in tokens)
def _build_byte_map() -> Tuple[Dict[int, str], Dict[str, int]]:
    nice = (
        list(range(ord('!'), ord('~') + 1)) +
        list(range(ord('¡'), ord('¬') + 1)) +
        list(range(ord('®'), ord('ÿ') + 1))
    )
    b2c, n = {}, 256
    for b in range(256):
        b2c[b] = chr(b) if b in nice else (lambda: chr(n))()
        if b not in nice: n += 1
    # rebuild cleanly
    b2c, n = {}, 256
    for b in range(256):
        if b in nice:
            b2c[b] = chr(b)
        else:
            b2c[b] = chr(n); n += 1
    return b2c, {v: k for k, v in b2c.items()}


_B2C, _C2B = _build_byte_map()


def _word_to_chars(word: str) -> Tuple[str, ...]:
    return tuple(_B2C[b] for b in word.encode('utf-8'))


def _get_pairs(vocab: Dict[Tuple, int]) -> Dict[Tuple[str, str], int]:
    """Count all adjacent symbol pairs across all words."""
    counts: Dict[Tuple[str, str], int] = collections.defaultdict(int)
    for symbols, freq in vocab.items():
        for i in range(len(symbols) - 1):
            counts[(symbols[i], symbols[i + 1])] += freq
    return counts


def _merge_pair(
    vocab  : Dict[Tuple, int],
    pair   : Tuple[str, str],
) -> Dict[Tuple, int]:
    """Return new vocab with every occurrence of pair merged into one symbol."""
    a, b   = pair
    merged = a + b
    new_vocab: Dict[Tuple, int] = {}
    for symbols, freq in vocab.items():
        out, i = [], 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged); i += 2
            else:
                out.append(symbols[i]); i += 1
        new_vocab[tuple(out)] = freq
    return new_vocab


def build_vocabulary(corpus_texts: List[str]) -> Tuple[dict, list]:
    """
    Run byte-level BPE on corpus_texts until TARGET_VOCAB_SIZE is reached.
    Returns (token→id dict, list of merge pairs).
    """
    print(f"  Building vocabulary from {len(corpus_texts):,} documents …")

    # Count word frequencies
    word_freq: Dict[str, int] = collections.defaultdict(int)
    for doc in corpus_texts:
        for word in _PRESPLIT.findall(doc):
            word_freq[word] += 1

    # Initialise vocab: each word is a sequence of byte-chars
    bpe_vocab: Dict[Tuple, int] = {
        _word_to_chars(w): f for w, f in word_freq.items() if f > 0
    }

    # Initial token set = all unique byte-chars  (256 base tokens)
    token_set: set = set()
    for symbols in bpe_vocab:
        token_set.update(symbols)

    merge_rules: List[Tuple[str, str]] = []

    while len(token_set) < TARGET_VOCAB_SIZE:
        pairs = _get_pairs(bpe_vocab)
        if not pairs:
            break
        best  = max(pairs, key=pairs.get)
        bpe_vocab   = _merge_pair(bpe_vocab, best)
        merge_rules.append(best)
        token_set.add(best[0] + best[1])

        if len(token_set) % 5000 == 0:
            print(f"    vocab size: {len(token_set):,} / {TARGET_VOCAB_SIZE:,}")

    # Assign integer ids
    sorted_tokens = sorted(token_set)
    tok2id = {t: i for i, t in enumerate(sorted_tokens)}
    print(f"  Vocabulary built: {len(tok2id):,} tokens, {len(merge_rules):,} merge rules")
    return tok2id, merge_rules


def save_vocab_files(
    tok2id      : dict,
    merge_rules : list,
    vocab_path  : str = 'vocab.json',
    merges_path : str = 'bpe_merges.txt',
):
    with open(vocab_path, 'w', encoding='utf-8') as fh:
        json.dump(tok2id, fh, ensure_ascii=False, indent=2)
    with open(merges_path, 'w', encoding='utf-8') as fh:
        for a, b in merge_rules:
            fh.write(f'{a} {b}\n')
    print(f"  ✓ vocab  → {vocab_path!r}  ({len(tok2id):,} tokens)")
    print(f"  ✓ merges → {merges_path!r}  ({len(merge_rules):,} rules)")


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2  –  Transformer model + pre-training loop
# ═══════════════════════════════════════════════════════════════════════════════

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    ctx_len    : int   = 1024       # context window length
    vocab_size : int   = TARGET_VOCAB_SIZE
    n_layers   : int   = 12
    n_heads    : int   = 12
    d_model    : int   = 768
    dropout    : float = 0.1

TOTAL_STEPS   = 300_000
WARMUP_STEPS  = 2_000
BATCH_SIZE    = 16
LR_MAX        = 3e-4
LR_MIN        = 1e-4
WEIGHT_DECAY  = 0.1
GRAD_CLIP     = 1.0
SAVE_EVERY    = 5_000
LOG_EVERY     = 100
CHECKPOINT    = 'model_weights.pt'

# ── Architecture ──────────────────────────────────────────────────────────────

class SelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.c_attn  = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj  = nn.Linear(cfg.d_model, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len))
                  .view(1, 1, cfg.ctx_len, cfg.ctx_len)
        )

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.n_heads
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        q = q.view(B, T, self.n_heads, hs).transpose(1, 2)
        k = k.view(B, T, self.n_heads, hs).transpose(1, 2)
        v = v.view(B, T, self.n_heads, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = self.drop(F.softmax(att, dim=-1))
        y   = att @ v
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.act    = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model)
        self.drop   = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = SelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp  = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.d_model),
            wpe  = nn.Embedding(cfg.ctx_len,    cfg.d_model),
            drop = nn.Dropout(cfg.dropout),
            h    = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            ln_f = nn.LayerNorm(cfg.d_model),
        ))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight   # weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        x    = self.transformer.drop(
                   self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Streaming dataset ─────────────────────────────────────────────────────────

class WebCorpusStream(IterableDataset):
    def __init__(self, tok2id: dict, merge_rules: list, ctx: int):
        from datasets import load_dataset
        self.ds    = load_dataset('openwebtext', split='train', streaming=True)
        self.tok2id = tok2id
        self.ranks  = {(a, b): r for r, (a, b) in enumerate(merge_rules)}
        self.ctx    = ctx

    def _encode(self, text: str) -> list:
        ids = []
        for word in _PRESPLIT.findall(text):
            chars = _word_to_chars(word)
            # apply merges inline
            syms  = chars
            while True:
                ps   = {(syms[i], syms[i+1]) for i in range(len(syms)-1)}
                if not ps: break
                best = min(ps, key=lambda p: self.ranks.get(p, float('inf')))
                if best not in self.ranks: break
                a, b = best
                out, i = [], 0
                while i < len(syms):
                    if i < len(syms)-1 and syms[i]==a and syms[i+1]==b:
                        out.append(a+b); i += 2
                    else:
                        out.append(syms[i]); i += 1
                syms = tuple(out)
            ids.extend(self.tok2id.get(s, 0) for s in syms)
        return ids

    def __iter__(self):
        buf = []
        for sample in self.ds:
            buf.extend(self._encode(sample['text']))
            while len(buf) >= self.ctx + 1:
                chunk = buf[:self.ctx + 1]
                yield (torch.tensor(chunk[:-1], dtype=torch.long),
                       torch.tensor(chunk[1:],  dtype=torch.long))
                buf = buf[self.ctx:]


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR_MAX * step / WARMUP_STEPS
    p = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * p))


# ── Checkpoint ────────────────────────────────────────────────────────────────

def save_checkpoint(model: LanguageModel, step: int):
    sd = {k: v.cpu().contiguous()
          for k, v in model.state_dict().items()
          if not k.endswith('.bias')}
    torch.save(sd, CHECKPOINT)
    print(f"  [step {step:,}] checkpoint saved → '{CHECKPOINT}'")


# ── Pre-training loop ─────────────────────────────────────────────────────────

def pretrain(tok2id: dict, merge_rules: list):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg    = ModelConfig()
    model  = LanguageModel(cfg).to(device)
    print(f"\n── Pre-training ────────────────────────────────────────")
    print(f"Parameters : {model.num_params()/1e6:.1f}M")
    print(f"Device     : {device}  |  Steps: {TOTAL_STEPS:,}  |  Batch: {BATCH_SIZE}")

    dataset = WebCorpusStream(tok2id, merge_rules, cfg.ctx_len)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE)

    decay   = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_dec  = [p for n, p in model.named_parameters() if p.dim() <  2]
    optim   = torch.optim.AdamW([
        {'params': decay,  'weight_decay': WEIGHT_DECAY},
        {'params': no_dec, 'weight_decay': 0.0},
    ], lr=LR_MAX, betas=(0.9, 0.95), eps=1e-8)
    scaler  = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    step, t0, loss_acc = 0, time.time(), 0.0
    for xb, yb in loader:
        if step >= TOTAL_STEPS: break
        xb, yb = xb.to(device), yb.to(device)
        for pg in optim.param_groups:
            pg['lr'] = get_lr(step)
        with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
            _, loss = model(xb, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optim); scaler.update()
        optim.zero_grad(set_to_none=True)
        loss_acc += loss.item()
        step     += 1
        if step % LOG_EVERY == 0:
            print(f"  step {step:>7,}  loss={loss_acc/LOG_EVERY:.4f}"
                  f"  lr={get_lr(step):.2e}  "
                  f"elapsed={( time.time()-t0)/60:.1f}min")
            loss_acc = 0.0
        if step % SAVE_EVERY == 0:
            save_checkpoint(model, step)

    save_checkpoint(model, step)
    print(f"\n✓ Pre-training complete → '{CHECKPOINT}'")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("═" * 60)
    print("Phase 1 : Building BPE vocabulary from OpenWebText …")
    print("═" * 60)
    from datasets import load_dataset
    raw_ds = load_dataset('openwebtext', split='train', streaming=True)
    corpus_sample = []
    for i, doc in enumerate(raw_ds):
        corpus_sample.append(doc['text'])
        if i >= CORPUS_SAMPLE: break

    tok2id, merge_rules = build_vocabulary(corpus_sample)
    save_vocab_files(tok2id, merge_rules)

    print("\n" + "═" * 60)
    print("Phase 2 : Pre-training transformer …")
    print("═" * 60)
    pretrain(tok2id, merge_rules)