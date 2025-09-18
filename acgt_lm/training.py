import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from misc import *
from model import *

D_MODEL = 10
N_LAYERS = 1
N_HEAD = 1

BLOCK_SIZE = 100 # context window
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
N_EPOCHS = 3000
SEED = 314

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_prepare_data(data_path):
    df = pd.read_pickle(data_path)
    sequences = df["sequence"].tolist()
    encoded = [tok for s in sequences for tok in encode(s)]
    data = torch.tensor(encoded, dtype=torch.long)

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    return train_data, test_data

def get_batch(split, batch_size=BATCH_SIZE, seq_len=BLOCK_SIZE):
    src = train_data if split == "train" else test_data
    ix = torch.randint(0, len(src) - seq_len - 1, (batch_size,))

    x = torch.stack([src[i : i + seq_len] for i in ix]).to(device)
    y = torch.stack([src[i + 1 : i + seq_len + 1] for i in ix]).to(device)

    return x, y

def evaluate():
    model.eval()
    with torch.no_grad():
        xb, yb = get_batch("test")
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))

    return loss.item()

def train(model, optimizer):
    train_losses, val_losses, steps = [], [], []
    best_val_loss = float("inf")
    for step in tqdm(range(N_EPOCHS), desc="Training"):
        model.train()
        xb, yb = get_batch("train")
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        val_loss = evaluate()
        train_losses.append(loss.item())
        val_losses.append(val_loss)
        steps.append(step)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    return train_losses, val_losses, steps

if __name__ == '__main__':

    train_data, test_data = load_and_prepare_data("data.pkl")
    model = ACGT_LM(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEAD, block_size=BLOCK_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses, steps = train(model, optimizer)

    data = {
        "Step": steps,
        "Train Loss": train_losses,
        "Validation Loss": val_losses
    }

    df = pd.DataFrame(data)
    df.to_csv("losses.csv", index=False)
