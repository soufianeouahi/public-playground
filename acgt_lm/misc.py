import torch
import torch.nn.functional as F

import pandas as pd

NUCLEOTIDES = ["A", "C", "G", "T"]
NUC_INDEX = {nuc: i for i, nuc in enumerate(NUCLEOTIDES)}

stoi = {nuc: idx for idx, nuc in enumerate(NUCLEOTIDES)}
itos = {idx: nuc for nuc, idx in stoi.items()}

vocab_size = len(NUCLEOTIDES)

# tokenization functions
def encode(seq): return [stoi[c] for c in seq]
def decode(ids): return ''.join(itos[i] for i in ids)

@torch.no_grad()
def generate_sequence(model, start_sequence, max_new_tokens, temperature=1.0):
    model.eval()
    device = next(model.parameters()).device

    idx = torch.tensor([encode(start_sequence)], dtype=torch.long).to(device)

    for _ in range(max_new_tokens):
        # crop context, call model, take last-step logits
        idx_cond = idx[:, -model.block_size:]
        logits = model(idx_cond)
        logits = logits[:, -1, :]

        logits = logits / temperature

        # sampling
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_token), dim=1)

    generated_sequence = decode(idx[0].tolist())

    return generated_sequence

def sample_sequences(model, n_generate=100, sequence_length=100, start_sequence="A", temperature=1.0):
    model.eval()
    generated_sequences = []

    for _ in range(n_generate):
        generated = generate_sequence(
            model,
            start_sequence=start_sequence,
            max_new_tokens=sequence_length - len(start_sequence),
            temperature=temperature,
        )

        generated_sequences.append(generated)

    return pd.DataFrame({'sequence': generated_sequences})