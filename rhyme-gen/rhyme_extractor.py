
with open('so-be-it.txt', encoding='utf-8') as f:
    lines = f.readlines()

clean_lines = list(dict.fromkeys(lines))


