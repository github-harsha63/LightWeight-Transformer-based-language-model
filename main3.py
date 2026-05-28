"""
main3.py  –  Fine-tuning
------------------------
Reads trainingdata.txt, fine-tunes the weights in model_weights.pt,
and writes the improved weights back to the same file.

Usage
-----
    python main3.py
    python main3.py --steps 200 --lr 1e-4
    python main3.py --data mytext.txt
"""

import math
import os
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from my_tokenizer import ByteTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    ctx_len    : int = 1024
    vocab_size : int = 50257
    n_layers   : int = 12
    n_heads    : int = 12
    d_model    : int = 768


# ─────────────────────────────────────────────────────────────────────────────
# Architecture  (key names must match model_weights.pt)
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.nh     = cfg.n_heads
        self.dm     = cfg.d_model
        self.c_attn = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.c_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(cfg.ctx_len, cfg.ctx_len))
                  .view(1, 1, cfg.ctx_len, cfg.ctx_len)
        )

    def forward(self, x):
        B, T, C = x.size()
        hs = C // self.nh
        q, k, v = self.c_attn(x).split(self.dm, dim=2)
        q = q.view(B, T, self.nh, hs).transpose(1, 2)
        k = k.view(B, T, self.nh, hs).transpose(1, 2)
        v = v.view(B, T, self.nh, hs).transpose(1, 2)
        w = F.softmax(
            (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
            .masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf')),
            dim=-1
        )
        return self.c_proj((w @ v).transpose(1, 2).contiguous().view(B, T, C))


class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.c_fc   = nn.Linear(cfg.d_model, 4 * cfg.d_model)
        self.act    = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * cfg.d_model, cfg.d_model)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Layer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.d_model)
        self.mlp  = FFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class LM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(cfg.vocab_size, cfg.d_model),
            wpe  = nn.Embedding(cfg.ctx_len,    cfg.d_model),
            h    = nn.ModuleList([Layer(cfg) for _ in range(cfg.n_layers)]),
            ln_f = nn.LayerNorm(cfg.d_model),
        ))
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        h = (self.transformer.wte(idx)
             + self.transformer.wpe(torch.arange(T, device=idx.device)))
        for layer in self.transformer.h:
            h = layer(h)
        logits = self.lm_head(self.transformer.ln_f(h))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ─────────────────────────────────────────────────────────────────────────────
# Weight I/O
# ─────────────────────────────────────────────────────────────────────────────

def load_weights(model: LM, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"'{path}' not found. Run main2.py first.")
    saved = torch.load(path, map_location='cpu', weights_only=True)
    miss, _ = model.load_state_dict(saved, strict=False)
    real_miss = [k for k in miss if not k.endswith('.bias')]
    if real_miss:
        print(f"[WARNING] missing: {real_miss}")
    print(f"  ✓ Loaded weights from '{path}'")


def save_weights(model: LM, path: str):
    sd = {k: v.cpu().contiguous()
          for k, v in model.state_dict().items()
          if not k.endswith('.bias')}
    torch.save(sd, path)
    print(f"  ✓ Saved weights → '{path}'")


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

class Corpus:
    def __init__(self, path: str, tok: ByteTokenizer, chunk: int, device: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' not found.")
        ids = tok.encode(open(path, encoding='utf-8').read())
        self.data   = torch.tensor(ids, dtype=torch.long, device=device)
        self.chunk  = chunk
        self.cursor = 0
        print(f"  ✓ Corpus: '{path}'  ({len(ids):,} tokens)")

    def batch(self, size: int):
        need = size * self.chunk + 1
        if self.cursor + need > len(self.data):
            self.cursor = 0
        s = self.data[self.cursor : self.cursor + need]
        self.cursor += size * self.chunk
        return s[:-1].view(size, self.chunk), s[1:].view(size, self.chunk)


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def fine_tune(
    weights_path : str   = 'model_weights.pt',
    data_path    : str   = 'trainingdata.txt',
    max_steps    : int   = 100,
    batch_size   : int   = 2,
    chunk_len    : int   = 256,
    lr_max       : float = 3e-4,
    lr_min       : float = 3e-5,
    save_every   : int   = 25,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n── Fine-tuning ─────────────────────────────────────────")
    print(f"Device  : {device}")
    print(f"Steps   : {max_steps}  |  batch={batch_size}  |  chunk={chunk_len}")

    tok    = ByteTokenizer()
    model  = LM(Config()).to(device)
    load_weights(model, weights_path)
    model.train()

    corpus = Corpus(data_path, tok, chunk_len, device)
    optim  = torch.optim.AdamW(model.parameters(), lr=lr_max, betas=(0.9, 0.95))

    for step in range(1, max_steps + 1):
        lr = lr_min + 0.5 * (lr_max - lr_min) * (
            1.0 + math.cos(math.pi * (step - 1) / max(1, max_steps - 1))
        )
        for pg in optim.param_groups:
            pg['lr'] = lr

        x, y = corpus.batch(batch_size)
        optim.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % max(1, max_steps // 10) == 0 or step == max_steps:
            print(f"  step {step:>5}/{max_steps}  loss={loss.item():.4f}  lr={lr:.2e}")

        if step % save_every == 0 or step == max_steps:
            save_weights(model, weights_path)

    print(f"\n✓ Done. Weights updated in '{weights_path}'")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights',    default='model_weights.pt')
    ap.add_argument('--data',       default='trainingdata.txt')
    ap.add_argument('--steps',      type=int,   default=100)
    ap.add_argument('--batch',      type=int,   default=2)
    ap.add_argument('--chunk',      type=int,   default=256)
    ap.add_argument('--lr',         type=float, default=3e-4)
    ap.add_argument('--lr-min',     type=float, default=3e-5)
    ap.add_argument('--save-every', type=int,   default=25)
    args = ap.parse_args()

    fine_tune(
        weights_path = args.weights,
        data_path    = args.data,
        max_steps    = args.steps,
        batch_size   = args.batch,
        chunk_len    = args.chunk,
        lr_max       = args.lr,
        lr_min       = args.lr_min,
        save_every   = args.save_every,
    )