from features import extract_features

dummy_pairs = [
    ("i walk alone <eol>",            "i stand strong <eol>"),
    ("the night is dark <eol>",       "but hope still shines <eol>"),
    ("beat drops heavy <eol>",        "flow stays steady <eol>"),
]

feats, rhyme_map = extract_features(dummy_pairs)
print('-' * 20)
print("Full rhyme_map:", rhyme_map)
print('-' * 20)
print(feats)