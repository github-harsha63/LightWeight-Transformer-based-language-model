# LightWeight Transformer-Based Language Model

A decoder-only transformer language model with a custom byte-level BPE tokenizer, built entirely from scratch in Python and PyTorch — no external tokenizer libraries, no hosted APIs, fully offline after setup.

> Heavily inspired by [Andrej Karpathy's](https://github.com/karpathy) work on minimal language model implementations.

---

## Features

- Custom Byte-Pair Encoding tokenizer implemented from scratch
- 12-layer decoder-only transformer with multi-head causal self-attention
- Autoregressive text generation via top-k sampling with configurable temperature
- Fine-tuning pipeline with cosine LR scheduling and gradient norm clipping
- Fully offline after one-time setup — no runtime internet dependency

---

## Project Structure

```
├── my_tokenizer.py       # Custom BPE tokenizer (no tiktoken, no HuggingFace)
├── main.py               # Generate text from a prompt (run this)
├── main3.py              # Fine-tune on custom text corpus
├── main2.py              # Pre-training pipeline reference
├── trainingdata.txt      # Domain fine-tuning corpus
└── download_weights.py   # One-time setup script
```

---

## Setup

**1. Clone**
```bash
git clone https://github.com/github-harsha63/LightWeight-Transformer-based-language-model.git
cd LightWeight-Transformer-based-language-model
```

**2. Install dependencies**
```bash
pip install torch transformers
```

**3. Download tokenizer files and model weights**
```bash
python download_weights.py
```

This downloads the GPT-2 vocabulary and BPE merge rules and saves them as `vocab.json` and `bpe_merges.txt`. Delete the script after running.

> **Model weights (`model_weights.pt`)** are not included in this repository due to file size constraints.
> The weights used in this project have been trained and evaluated independently.
> To request access to the pre-trained weights, contact: **harshavemula55@gmail.com**

---

## Usage

**Generate text**
```bash
python main.py "The future of artificial intelligence is"
```

**Options**
```bash
python main.py "Your prompt here" --seqs 3 --tokens 60 --topk 50 --temp 0.9
```

| Flag | Default | Description |
|---|---|---|
| `--seqs` | 3 | Number of sequences to generate |
| `--tokens` | 50 | New tokens per sequence |
| `--topk` | 50 | Top-k sampling cutoff |
| `--temp` | 1.0 | Sampling temperature |
| `--seed` | 42 | Random seed |

**Fine-tune on your own text**
```bash
python main3.py --steps 100 --data trainingdata.txt
```

---

## Model Architecture

| Parameter | Value |
|---|---|
| Layers | 12 |
| Attention Heads | 12 |
| Embedding Dimension | 768 |
| Context Window | 1024 tokens |
| Vocabulary Size | 50,257 |
| Parameters | ~124M |

---

## Authors

 Name : Vemula Harshavardhan Reddy 


**Guide:** Dr. LNC Prakash K, Associate Professor
**Institution:** CVR College of Engineering
**Department:** CSE (Data Science)
**Year:** 2025–2026

---

## Acknowledgements

This project draws significant inspiration from [Andrej Karpathy](https://github.com/karpathy) and his educational work on building language models from the ground up.

---

## Contact

For weight file access or any queries: **harshavemula55@gmail.com**
