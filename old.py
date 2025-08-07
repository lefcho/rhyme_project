import os
import glob
import string
import pronouncing


def find_lyrics_files(base_dir: str) -> list:
    """
    Recursively find all .txt lyric files under base_dir (organized by album folders).

    Args:
        base_dir: Path to the 'albums' directory.
    Returns:
        List of filepaths to all .txt files.
    """
    pattern = os.path.join(base_dir, '*', '*.txt')
    return glob.glob(pattern)


def load_song_lines(filepath: str, max_words: int = 30) -> list:
    """
    Read a song file and return a list of cleaned, deduplicated lyric lines,
    each terminated by an <eol> token.

    - Strips whitespace
    - Skips empty lines and annotations (e.g., [Verse])
    - Removes exact duplicate lines, preserving first occurrence order
    - Appends ' <eol>' to mark end-of-line boundaries
    - Skips lines longer than `max_words`

    Args:
        filepath: Path to a .txt lyric file.
        max_words: Maximum number of words allowed per line.
    Returns:
        List of unique lyric lines (lowercased, each ending with '<eol>').
    """
    seen = set()
    unique_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line or (line.startswith('[') and line.endswith(']')):
                continue
            line = line.lower()
            if line in seen:
                continue
            if len(line.split()) > max_words:
                continue
            seen.add(line)
            unique_lines.append(f"{line} <eol>")
    return unique_lines


def build_line_pairs(base_dir: str) -> list:
    """
    Traverse all lyrics and build (prev_line, next_line) pairs within each song.

    Args:
        base_dir: Path to the 'albums' directory.
    Returns:
        List of tuples (prev_line, next_line).
    """
    pairs = []
    files = find_lyrics_files(base_dir)
    for fp in files:
        song_lines = load_song_lines(fp)
        for i in range(len(song_lines) - 1):
            prev_line = song_lines[i]
            next_line = song_lines[i + 1]
            pairs.append((prev_line, next_line))
    return pairs


# -----------------------------
# Feature Extraction

def extract_features(pairs):
    """
    For each line pair, compute two features for the previous line:
      1. syllable_count: total syllables in the previous line
      2. rhyme_id: an integer encoding of the last word's rhyming part

    Args:
        pairs: list of (prev_line, next_line)
    Returns:
        features: list of (syllable_count, rhyme_id)
        rhyme_map: dict mapping rhyme_part -> id
    """
    rhyme_map = {}
    features = []
    next_rhyme_id = 0
    for prev, _ in pairs:
        tokens = prev.split()
        # syllable count
        syllables = 0
        for w in tokens:
            phones = pronouncing.phones_for_word(w)
            if phones:
                syllables += pronouncing.syllable_count(phones[0])
        # rhyme part of last word
        last = tokens[-2] if tokens[-1] == '<eol>' else tokens[-1]
        rp = pronouncing.rhyming_part(last) or last
        if rp not in rhyme_map:
            rhyme_map[rp] = next_rhyme_id
            next_rhyme_id += 1
        features.append((syllables, rhyme_map[rp]))
    return features, rhyme_map


# -----------------------------
# Tokenization and Vocabulary

def build_vocab(lines, min_freq=1):
    from collections import Counter
    freq = Counter(w for line in lines for w in line)
    vocab = [w for w, c in freq.items() if c >= min_freq]
    specials = ['<pad>', '<unk>']
    idx2word = specials + sorted(vocab)
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


# -----------------------------
# Preparing Dataset

def prepare_dataset(pairs, features, word2idx):
    """
    Convert line pairs and their features into input, feature, and target sequences.

    Args:
        pairs: List of (prev_line, next_line) strings.
        features: List of (syllable_count, rhyme_id) tuples.
        word2idx: Vocabulary mapping.
    Returns:
        inputs: list of padded input index lists
        feats: list of feature tuples
        targets: list of padded target index lists
    """
    tokenized = [(p.split(), n.split()) for p, n in pairs]
    max_in = max(len(t1) for t1, _ in tokenized)
    max_out = max(len(t2) for _, t2 in tokenized)

    inputs, feats_list, targets = [], [], []
    for (t1, t2), feat in zip(tokenized, features):
        idx1 = [word2idx.get(w, word2idx['<unk>']) for w in t1]
        idx2 = [word2idx.get(w, word2idx['<unk>']) for w in t2]
        idx1 += [word2idx['<pad>']] * (max_in - len(idx1))
        idx2 += [word2idx['<pad>']] * (max_out - len(idx2))
        inputs.append(idx1)
        feats_list.append(feat)
        targets.append(idx2)
    return inputs, feats_list, targets


# -----------------------------
# Example usage
if __name__ == '__main__':
    base = 'albums'
    pairs = build_line_pairs(base)
    features, rhyme_map = extract_features(pairs)
    all_lines = [l.split() for pair in pairs for l in pair]
    word2idx, idx2word = build_vocab(all_lines, min_freq=2)
    inputs, feats, targets = prepare_dataset(pairs, features, word2idx)
    print(f"Examples: {len(inputs)}")
    print(f"Sample input idxs: {inputs[0]}")
    print(f"Sample features: {feats[0]}")
    print(f"Sample target idxs: {targets[0]}")
