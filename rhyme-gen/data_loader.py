import os
import glob
import re


def clean_line_tokens(line: str) -> str:
    """
    Remove leading/trailing non-word chars from each token.
    """
    tokens = line.split()
    cleaned = []
    for tok in tokens:
        # Strip non-alphanumeric at start/end
        tok = re.sub(r'^[^\w]+|[^\w]+$', '', tok)
        if tok:
            cleaned.append(tok)
    return ' '.join(cleaned)


def find_lyrics_files(base_dir: str) -> list:
    """
    Extract all file paths.
    """
    pattern = os.path.join(base_dir, '*', '*.txt')
    return glob.glob(pattern)


def load_song_lines(filepath: str, max_words: int = 30) -> list:
    """
    Read a song file and return a list of cleaned, deduplicated lyric lines,
    each ending with an <eol> token.
    """
    seen = set()
    unique_lines = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            # skip blanks and bracketed annotations
            if not line or (line.startswith('[') and line.endswith(']')):
                continue
            line = line.lower()
            line = clean_line_tokens(line)
            if line in seen:
                continue

            if len(line.split()) > max_words:
                continue
            seen.add(line)
            
            # append end-of-line token
            unique_lines.append(f"{line} <eol>")
    return unique_lines

def build_line_pairs(base_dir: str) -> list:
    """
    Traverse all lyrics and build (prev_line, next_line) 
    pairs within each song.
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


def prepare_dataset(pairs, word2idx):
    """
    Convert line pairs into input and target index sequences, 
    padded to max length (equal size).
    """
    tokenized = [(line1.split(), line2.split()) for line1, line2 in pairs]
    max_lenght = max(len(t1) for t1, _ in tokenized)

    inputs, targets = [], []
    for t1, t2 in tokenized:
        # map to idx, pad
        idx1 = [word2idx.get(w, word2idx['<unk>']) for w in t1]
        idx2 = [word2idx.get(w, word2idx['<unk>']) for w in t2]
        # pad with <pad> token (0)
        idx1 += [word2idx['<pad>']] * (max_lenght - len(idx1))
        idx2 += [word2idx['<pad>']] * (max_lenght - len(idx2))

        inputs.append(idx1)
        targets.append(idx2)
    
    return inputs, targets


# Example usage
if __name__ == '__main__':
    base_folder = 'albums'
    pairs = build_line_pairs(base_folder)
    print(f"Total line pairs: {len(pairs)}")
    # Vocabulary and dataset
    from vocabulary import build_vocab
    word2idx, idx2word = build_vocab(pairs)
    inputs, targets = prepare_dataset(pairs, word2idx)
    print(f"Dataset sizes -> inputs: {len(inputs)}, targets: {len(targets)}")
