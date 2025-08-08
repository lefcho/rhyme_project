import pronouncing


def get_rm_part(word):
    """
    Extract the rhyming part of the word.
    """
    phones = pronouncing.phones_for_word(word)
    if not phones:
        rm_part = word
    else:
        phone_seq = phones[0]
        rm_part = pronouncing.rhyming_part(phone_seq) or phone_seq
    
    return rm_part
 


def syllable_count(line: str) -> int:
    """
    Count total syllables in a line.
    """
    total = 0

    for w in line.split():
        phones = pronouncing.phones_for_word(w)
        if phones:
            total += pronouncing.syllable_count(phones[0])

    return total


def build_rhyme_map(lines: list) -> dict:
    """
    Given a list of lines, map each unique rhyming_part to an integer ID.
    """
    rhyme_map = {}
    next_id = 0
    for line in lines:
        tokens = line.split()
        last = tokens[-2] if tokens[-1] == '<eol>' else tokens[-1]
        rm_part = get_rm_part(last)

        if rm_part not in rhyme_map:
            rhyme_map[rm_part] = next_id
            next_id += 1
    return rhyme_map


def rhyme_id(line: str, rhyme_map: dict) -> int:
    """
    Return the integer ID of the line's last word's rhyming part.
    """
    tokens = line.split()
    last = tokens[-2] if tokens[-1] == '<eol>' else tokens[-1]

    rm_part = get_rm_part(last)

    return rhyme_map.get(rm_part, -1)


def extract_features(pairs: list):
    """
    For each pair, compute (syllable_count, rhyme_id) for prev.
    """
    prev_lines = [p for p, _ in pairs]
    rhyme_map = build_rhyme_map(prev_lines)

    feats = []
    for prev, _ in pairs:
        feats.append((syllable_count(prev), rhyme_id(prev, rhyme_map)))
    
    return feats, rhyme_map

# print(syllable_count('And the drought will define a man when the well dries up <eol>'))
# build_rhyme_map(["'Cause I'm still paranoid to this running", "And it's nobody fault, I made the decisions I made"])

