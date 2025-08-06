import pronouncing
import string


# Load and clean lyrics
def load_lyrics(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return list(dict.fromkeys(lines))  # Remove exact duplicate lines

# Normalize and tokenize lyrics
def tokenize_lyrics(lines):
    word_list = []

    for i, line in enumerate(lines):
        for word in line.split():
            word_clean = word.strip(string.punctuation).lower()
            if word_clean:
                word_list.append((word_clean, i))

    return word_list

# Get vowel phonemes from CMU dict
def vowels_of_word(word):
    phones_list = pronouncing.phones_for_word(word)

    if not phones_list:
        return []
    
    phones = phones_list[0].split()
    vowels = [ph for ph in phones if ph[-1].isdigit()]

    return vowels

# Find longest matching vowel sequence from the end
def longest_common_suffix(seq1, seq2):
    match_len = 0

    for v1, v2 in zip(reversed(seq1), reversed(seq2)):
        if v1 == v2:
            match_len += 1
        else:
            break
    return match_len

# Compute rhyme density
def compute_rhyme_density(word_list, word_vowels, window=2):
    longest_matches = []

    for i, (word, line_idx) in enumerate(word_list):
        seq1 = word_vowels.get(word, [])
        if not seq1:
            continue

        best_len = 0
        best_partner = None

        # scan nearby words
        for j, (word2, line_j) in enumerate(word_list):
            if i == j or word == word2:
                continue
            if abs(line_j - line_idx) > window:
                continue

            seq2 = word_vowels.get(word2, [])
            if not seq2:
                continue

            common = longest_common_suffix(seq1, seq2)
            if common > best_len:
                best_len = common
                best_partner = word2

        # only count multi-syllable rhymes
        if best_len >= 2 and best_partner:
            print(f"-> {word} rhymes ({best_len} vowels) with {best_partner}")
            longest_matches.append(best_len)

    # average of all best rhyme-lengths
    if longest_matches:
        return sum(longest_matches) / len(longest_matches)
    else:
        return 0

# Master function
def analyze_lyrics_rhyme_density(filepath):
    lines = load_lyrics(filepath)
    word_list = tokenize_lyrics(lines)
    unique_words = {w for w, _ in word_list}
    word_vowels = {w: vowels_of_word(w) for w in unique_words}
    # print(word_list)
    # print('-----------------------')
    # print(word_vowels)
    return compute_rhyme_density(word_list, word_vowels, window=2)

# Example usage
if __name__ == "__main__":
    rhyme_density = analyze_lyrics_rhyme_density("so-be-it.txt")
    print(f"Rhyme Density: {rhyme_density:.3f}")
