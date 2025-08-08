import json
import pickle


def load_mappings(vocab_path="vocab.json", rhyme_path="rhyme_map.pkl"):
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        word2idx = data["word2idx"]
        idx2word = data["idx2word"]

    with open(rhyme_path, "rb") as f:
        rhyme_map = pickle.load(f)

    return word2idx, idx2word, rhyme_map

word2idx, idx2word, rhyme_map = load_mappings()

line = [5606, 4700, 5510, 2498, 89]
rhyme = 0

for w in line:
    print(idx2word[w])
