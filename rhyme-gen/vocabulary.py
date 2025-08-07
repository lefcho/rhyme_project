from collections import Counter


def build_vocab(lines, min_freq=1):
    SPECIAL_TOKENS = ['<pad>', '<unk>']
    """
    Build word-to-index and index-to-word mappings from a list of tokenized lines.
    """
    freq = Counter()
    for line in lines:
        for w in line:
            freq[w] += 1

    vocab = [w for w, c in freq.items() if c >= min_freq]
    # reserve special tokens
    words = SPECIAL_TOKENS + sorted(vocab)
    word2idx = {w: i for i, w in enumerate(words)}
    print(word2idx)
    return word2idx, words


def tokenize_line(line, word2idx):
    """Convert a single line string into a list of token indices."""
    tokens = line.split()
    return [word2idx.get(tok, word2idx['<unk>']) for tok in tokens]

