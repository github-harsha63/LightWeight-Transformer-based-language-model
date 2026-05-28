"""
main.py  –  Generate
--------------------
Load weights from model_weights.pt, encode a prompt with the local
tokenizer, run autoregressive sampling, and print the result.

Usage
-----
    python main.py "Once upon a time"
    python main.py "The theory of gravity" --seqs 5 --tokens 80
    python main.py "In the beginning"      --topk 40 --temp 0.8
"""

import sys
import math
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from my_tokenizer import ByteTokenizer

# Config
@dataclass
class Config:
    ctx_len = 1024
    vocab_size = 50257
    n_layers = 12
    n_heads = 12
    d_model = 768


# Architecture  (key names must match model_weights.pt)
class Attention(nn.Module):
    def __init__(self, cfg):
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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1) @ v
        return self.c_proj(att.transpose(1, 2).contiguous().view(B, T, C))


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

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device)
        h    = self.transformer.wte(idx) + self.transformer.wpe(pos)
        for layer in self.transformer.h:
            h = layer(h)
        return self.lm_head(self.transformer.ln_f(h))


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_model(path: str, device: str) -> LM:
    import os
    if not os.path.exists(path):
        sys.exit(f"[ERROR] '{path}' not found. Run main2.py first.")
    model = LM(Config()).to(device)
    saved = torch.load(path, map_location=device, weights_only=True)
    miss, _ = model.load_state_dict(saved, strict=False)
    real_miss = [k for k in miss if not k.endswith('.bias')]
    if real_miss:
        print(f"[WARNING] missing keys: {real_miss}", file=sys.stderr)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model      : LM,
    tok        : ByteTokenizer,
    prompt     : str,
    num_seqs   : int   = 3,
    max_new    : int   = 50,
    top_k      : int   = 50,
    temperature: float = 1.0,
    seed       : int   = 42,
    device     : str   = 'cpu',
) -> list:
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)

    prompt_ids  = tok.encode(prompt)
    ids = (
        torch.tensor(prompt_ids, dtype=torch.long, device=device)
             .unsqueeze(0).repeat(num_seqs, 1)
    )

    while ids.size(1) < len(prompt_ids) + max_new:
        ctx    = ids[:, -model.cfg.ctx_len:]
        logits = model(ctx)[:, -1, :] / temperature

        if top_k > 0:
            top_vals, _ = torch.topk(logits, top_k, dim=-1)
            logits      = logits.masked_fill(logits < top_vals[:, -1:], float('-inf'))

        next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
        ids     = torch.cat((ids, next_id), dim=1)

    return [tok.decode(ids[i].tolist()) for i in range(num_seqs)]


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Generate text from a trained language model')
    ap.add_argument('prompt',          type=str,
                    help='Starting text prompt (wrap in quotes)')
    ap.add_argument('--weights',       default='model_weights.pt')
    ap.add_argument('--seqs',          type=int,   default=3,
                    help='Number of sequences  (default: 3)')
    ap.add_argument('--tokens',        type=int,   default=50,
                    help='New tokens per sequence  (default: 50)')
    ap.add_argument('--topk',          type=int,   default=50,
                    help='Top-k sampling  (default: 50)')
    ap.add_argument('--temp',          type=float, default=1.0,
                    help='Temperature  (default: 1.0)')
    ap.add_argument('--seed',          type=int,   default=42)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model(args.weights, device)
    model.eval()
    tok    = ByteTokenizer()

    outputs = generate(
        model       = model,
        tok         = tok,
        prompt      = args.prompt,
        num_seqs    = args.seqs,
        max_new     = args.tokens,
        top_k       = args.topk,
        temperature = args.temp,
        seed        = args.seed,
        device      = device,
    )

    print(f'\nPrompt: {args.prompt!r}\n')
    for i, text in enumerate(outputs, 1):
        print(f'[{i}] {text}\n')


if __name__ == '__main__':
    main()