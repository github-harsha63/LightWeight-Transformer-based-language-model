import os, json, shutil, tempfile, torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

TRANSPOSED = [
    'attn.c_attn.weight', 'attn.c_proj.weight',
    'mlp.c_fc.weight',    'mlp.c_proj.weight',
]

def save_tokenizer_files():
    tok = GPT2TokenizerFast.from_pretrained('gpt2')
    with tempfile.TemporaryDirectory() as tmp:
        tok.save_pretrained(tmp)
        # read tokenizer.json which contains everything
        with open(os.path.join(tmp, 'tokenizer.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)
    # extract vocab
    vocab = data['model']['vocab']
    with open('vocab.json', 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    # extract merges
    merges = data['model']['merges']
    with open('bpe_merges.txt', 'w', encoding='utf-8') as f:
        for pair in merges:
            f.write(' '.join(pair) + '\n')
    print("done: vocab.json + bpe_merges.txt")

def save_model_weights():
    sd = GPT2LMHeadModel.from_pretrained('gpt2').state_dict()
    sd = {k: v for k, v in sd.items()
          if not k.endswith('.attn.masked_bias')
          and not k.endswith('.attn.bias')}
    out = {}
    for k, v in sd.items():
        out[k] = v.t().contiguous() if any(k.endswith(s) for s in TRANSPOSED) else v.contiguous()
    torch.save(out, 'model_weights.pt')
    print("done: model_weights.pt")

if __name__ == '__main__':
    save_tokenizer_files()
    save_model_weights()
    print("All done. Delete this file now.")