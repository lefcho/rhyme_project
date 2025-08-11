import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import RapLinesDataset, collate_fn
from model import NextLineModel

CSV_PATH = "rap_lines_data.csv"
VOCAB_PATH = "vocab.json"
MODEL_OUT = "model.pth"

BATCH_SIZE = 32
EPOCHS = 10
LEARN_RATE = 1e-3


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)
word2idx = vocab["word2idx"]
vocab_size = len(word2idx)
pad_idx = word2idx.get("<pad>")

# Dataset / Loader
dataset = RapLinesDataset(CSV_PATH)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

# Model
model = NextLineModel(
    vocab_size=vocab_size,
    num_rhyme_ids=dataset.num_rhyme_ids,
    pad_idx=pad_idx,
).to(device)

# Losses
token_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

class_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_token_loss = 0.0
    total_class_loss = 0.0
    total_loss = 0.0

    for prev_batch, next_batch, syll_batch, rhyme_batch in loader:
        prev_batch = prev_batch.to(device)
        next_batch = next_batch.to(device)
        syll_batch = syll_batch.to(device) 
        rhyme_batch = rhyme_batch.to(device)

        optimizer.zero_grad()

        token_logits, rhyme_logits = model(prev_batch, next_batch, syll_batch, rhyme_batch)
        targets = next_batch[:, 1:].contiguous()

        token_loss = token_criterion(
            token_logits.view(-1, vocab_size),
            targets.view(-1),
        )
        class_loss = class_criterion(rhyme_logits, rhyme_batch)

        loss = token_loss + class_loss
        loss.backward()

        # make the norm 1 if higher
        nn.utils.clip_grad_norm_(model.parameters(), 1)

        optimizer.step()

        total_token_loss += token_loss.item()
        total_class_loss += class_loss.item()
        total_loss += loss.item()

    n_batches = len(loader)
    print(
        f"Epoch {epoch:02d} | "
        f"TokenLoss: {total_token_loss / n_batches:.4f} | "
        f"RhymeLoss: {total_class_loss / n_batches:.4f} | "
        f"Total: {total_loss / n_batches:.4f}"
    )

# Save
torch.save(model.state_dict(), MODEL_OUT)
print(f"Saved trained model to {MODEL_OUT}")
