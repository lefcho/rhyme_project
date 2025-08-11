import json
import pickle
import random
import torch
from model import NextLineModel
from dataset import RapLinesDataset
from features import syllable_count


# ----------------- Config -----------------
CSV_PATH = "rap_lines_data.csv"
VOCAB_PATH = "vocab.json"
RHYME_MAP_PATH = "rhyme_map.pkl"
MODEL_PATH = "model.pth"

MAX_TOKENS = 24  # safety cap
# ------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load vocab
with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab_data = json.load(f)
word2idx = vocab_data["word2idx"]
idx2word = {int(k): v for k, v in enumerate(vocab_data["idx2word"])}

PAD = word2idx.get("<pad>", 0)
EOL = word2idx.get("<eol>", None)

# load rhyme_map (token_idx -> rhyme_id)
with open(RHYME_MAP_PATH, "rb") as f:
    rhyme_map = pickle.load(f)

# dataset (for syllable normalization if you condition on it)
dataset = RapLinesDataset(CSV_PATH)
num_rhyme_ids = dataset.num_rhyme_ids
max_syllables = dataset.max_syllables

# model
vocab_size = len(word2idx)
model = NextLineModel(vocab_size=vocab_size, num_rhyme_ids=num_rhyme_ids, pad_idx=PAD).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

_TEMP = 0.9          # mild temperature
_TOP_K = 8           # sample from top-k for middle tokens
_TOP_P = 0.9         # nucleus cap
_RHYME_TOP_K = 5     # sample among top-k within rhyme set for the final word
_REP_PENALTY = 0.8   # discourage repeats

def apply_repetition_penalty(logits: torch.Tensor, freq: dict):
    if not freq:
        return logits
    logits = logits.clone()
    for tid, c in freq.items():
        if c > 0 and 0 <= tid < logits.numel():
            logits[tid] -= _REP_PENALTY * float(c)
    return logits

def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = _TOP_K, top_p: float = _TOP_P):
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        thresh = torch.topk(logits, k)[0][..., -1].unsqueeze(-1)
        logits = torch.where(logits < thresh, torch.tensor(float('-inf'), device=logits.device), logits)
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        mask = cumprobs > top_p
        mask[..., 0] = False
        filtered = torch.full_like(sorted_logits, float('-inf'))
        filtered[~mask] = sorted_logits[~mask]
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(0, sorted_idx, filtered)
    return logits

def sample_from_logits(logits: torch.Tensor, forbid_ids=None) -> int:
    if forbid_ids:
        logits = logits.clone()
        for tid in forbid_ids:
            if 0 <= tid < logits.numel():
                logits[tid] = float('-inf')
    logits = logits / _TEMP
    logits = top_k_top_p_filtering(logits, _TOP_K, _TOP_P)
    probs = torch.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1).item()
    return int(idx)

def pad_or_truncate_indices(indices, length=30, pad_idx=0):
    return indices[-length:] if len(indices) >= length else indices + [pad_idx] * (length - len(indices))

def target_rhyme_id_from_tokens(tokens):
    """Try to get rhyme id from the prompt's last valid token; scan backwards if needed."""
    for t in reversed(tokens):
        tid = word2idx.get(t, word2idx.get("<unk>", PAD))
        rid = rhyme_map.get(tid)
        if rid is not None:
            return int(rid)
    return None

def build_rhyme_mask_by_id(target_rhyme_id: int):
    """1.0 where token's rhyme_id == target; excludes PAD/EOL."""
    mask = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    if target_rhyme_id is None:
        return mask
    for tid in range(vocab_size):
        if tid == PAD or (EOL is not None and tid == EOL):
            continue
        rid = rhyme_map.get(tid)
        if rid is not None and int(rid) == target_rhyme_id:
            mask[tid] = 1.0
    return mask

def build_suffix_mask(last_word_text: str, min_len=3):
    """Fallback rhyme mask: words sharing last N letters with the prompt's last word (basic)."""
    mask = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    w = last_word_text.lower()
    w = "".join(ch for ch in w if ch.isalnum() or ch == "'")
    if len(w) < min_len:
        return mask
    suf = w[-min_len:]
    for tid in range(vocab_size):
        if tid == PAD or (EOL is not None and tid == EOL):
            continue
        tok = idx2word.get(tid, "").lower()
        tok_clean = "".join(ch for ch in tok if ch.isalnum() or ch == "'")
        if len(tok_clean) >= min_len and tok_clean.endswith(suf):
            mask[tid] = 1.0
    return mask

# ---- generation ----
def generate_next_line(raw_prev: str, top_k=5, max_tokens=MAX_TOKENS):  # top_k kept for compat; ignored
    tokens = raw_prev.lower().split()
    prev_indices = [word2idx.get(t, word2idx.get("<unk>", 0)) for t in tokens]
    prev_indices = pad_or_truncate_indices(prev_indices, length=30, pad_idx=PAD)
    prev_tensor = torch.tensor(prev_indices, dtype=torch.long, device=device).unsqueeze(0)

    # Encode previous line
    h_enc, c_enc = model.encode(prev_tensor)

    # Target rhyme id or fallback suffix mask
    target_rhyme = target_rhyme_id_from_tokens(tokens)
    rhyme_mask = build_rhyme_mask_by_id(target_rhyme)
    last_word_text = tokens[-1] if tokens else ""
    suffix_mask = build_suffix_mask(last_word_text, min_len=3) if rhyme_mask.sum() == 0 else None

    # Syllable budget with small jitter
    base_budget = syllable_count(raw_prev)
    base_budget = max(6, min(14, base_budget))
    syll_budget = max(5, min(16, base_budget + random.choice([-1, 0, 1])))
    syll_norm = float(syll_budget) / float(max_syllables if max_syllables > 0 else 1.0)

    # Condition hidden (rhyme + syllables)
    rhyme_tensor = torch.tensor([target_rhyme if target_rhyme is not None else 0],
                                dtype=torch.long, device=device)
    syll_tensor = torch.tensor([syll_norm], dtype=torch.float32, device=device)
    h_cond, c_cond = model._condition_hidden(rhyme_tensor, syll_tensor)
    hidden = (h_enc + h_cond, c_enc + c_cond)

    # Start decoding WITHOUT <sos>: use neutral <pad> as the first input
    input_token = torch.tensor([PAD], dtype=torch.long, device=device)

    generated, freq = [], {}
    syll_count = 0

    for step in range(max_tokens):
        logits, hidden = model.decode_step(input_token, hidden)
        logits = logits.squeeze(0)

        # never pick PAD; avoid early <eol>
        logits = logits.clone()
        logits[PAD] = float('-inf')
        if EOL is not None:
            logits[EOL] = float('-inf')

        # repetition penalty
        logits = apply_repetition_penalty(logits, freq)

        # ready to finish?
        finishing = (syll_count >= syll_budget - 1) or (len(generated) >= max_tokens - 1)

        if finishing:
            forced = logits.clone()
            used_mask = False
            if rhyme_mask is not None and rhyme_mask.sum() > 0:
                forced[rhyme_mask == 0] = float('-inf')
                used_mask = True
            elif suffix_mask is not None and suffix_mask.sum() > 0:
                forced[suffix_mask == 0] = float('-inf')
                used_mask = True

            if used_mask and not torch.isneginf(forced).all():
                forced = top_k_top_p_filtering(forced, top_k=_RHYME_TOP_K, top_p=1.0)
                probs = torch.softmax(forced / _TEMP, dim=-1)
                next_idx = int(torch.multinomial(probs, 1).item())
            else:
                next_idx = int(torch.argmax(logits).item())

            word = idx2word.get(next_idx, "<unk>")
            generated.append(word)
            break

        # middle tokens: sample for diversity
        next_idx = sample_from_logits(logits, forbid_ids=[PAD, EOL] if EOL is not None else [PAD])
        word = idx2word.get(next_idx, "<unk>")
        generated.append(word)
        input_token = torch.tensor([next_idx], dtype=torch.long, device=device)

        # update repetition counts and syllables
        freq[next_idx] = freq.get(next_idx, 0) + 1
        syll_count += syllable_count(word)

    return " ".join(generated)

if __name__ == "__main__":
    prompt = "All the static, shall cease to exist"
    out = generate_next_line(prompt)
    print("PROMPT:", prompt)
    print("GENERATED:", out)
