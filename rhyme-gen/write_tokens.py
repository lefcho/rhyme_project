import csv
from data_loader import build_line_pairs, prepare_dataset
from vocabulary import build_vocab
from features import extract_features
import json
import pickle
import csv


def write_to_csv(
    inputs: list,
    targets: list,
    syllables: list[int],
    rhyme_ids: list[int],
    output_path: str = "rap_lines_data.csv"
):
    """
    Write the processed rap‐line features to a CSV with columns:
      prev_indices, next_indices, syllables, rhyme_id
    Each of the index‐lists is joined as a space‐separated string.
    """
    assert len(inputs) == len(targets) == len(syllables) == len(rhyme_ids), \
        "All feature lists must be the same length"

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["prev_indices", "next_indices", "syllables", "rhyme_id"])

        for prev_idx, next_idx, syl, rid in zip(inputs, targets, syllables, rhyme_ids):
            prev_str = " ".join(map(str, prev_idx))
            next_str = " ".join(map(str, next_idx))
            writer.writerow([prev_str, next_str, syl, rid])

    print(f"Wrote {len(inputs)} examples to {output_path}")


def save_mappings(word2idx, idx2word, rhyme_map, vocab_path="vocab.json", rhyme_path="rhyme_map.pkl"):
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "word2idx": word2idx,
            "idx2word": idx2word
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to {vocab_path}")

    with open(rhyme_path, "wb") as f:
        pickle.dump(rhyme_map, f)
    print(f"Saved rhyme map to {rhyme_path}")


if __name__ == "__main__":
    base_folder = 'albums'
    pairs = build_line_pairs(base_folder)
    print(f"Total line pairs: {len(pairs)}")

    word2idx, idx2word = build_vocab(pairs)
    inputs, targets = prepare_dataset(pairs, word2idx)
    syllables, rhyme_ids, rhyme_map = extract_features(pairs)

    save_mappings(word2idx, idx2word, rhyme_map)
    write_to_csv(inputs, targets, syllables, rhyme_ids, output_path="rap_lines_data.csv")

