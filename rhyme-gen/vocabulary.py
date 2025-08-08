from collections import Counter


def build_vocab(pairs, min_freq=1):
    SPECIAL_TOKENS = ['<pad>', '<unk>']
    """
    Build word-to-index mappings from a list of lines.
    """
    freq = Counter()
    for pair in pairs:
        for w in pair[0].split():
            freq[w] += 1

    vocab = [w for w, c in freq.items() if c >= min_freq]
    # reserve special tokens
    words = SPECIAL_TOKENS + sorted(vocab)
    word2idx = {w: i for i, w in enumerate(words)}
    return word2idx, words



