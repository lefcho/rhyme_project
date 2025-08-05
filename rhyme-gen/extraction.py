# fetch_black_album_songs.py
# Fetch all songs from "The Black Album" by JAY-Z using the Genius API,
# remove section headers, skip non-song content, and save into a single text file.

import os
import lyricsgenius

# 2. Configure your Genius API token
GENIUS_API_TOKEN = "R5EeelIjRD8PXu6w1vxcYQk8TPGFTcrgDtP1HDHJgYEWZ73m88r5q0YAisX7ZRBS"
# Initialize Genius client with desired flags
genius = lyricsgenius.Genius(
    GENIUS_API_TOKEN,
    remove_section_headers=True,
    skip_non_songs=True,
    timeout=15,
    retries=3
)

# 3. Search the album
ARTIST_NAME = "JAY-Z"
ALBUM_NAME = "The Black Album"
# max_songs: upper bound on number of tracks (the album has 10)
album = genius.search_album(ALBUM_NAME, ARTIST_NAME)

# 4. Prepare output file
output_path = "the_black_album_lyrics.txt"
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    for idx, song in enumerate(album.songs, start=1):
        title = song.title.strip()
        lyrics = song.lyrics.strip()
        f.write(f"### {idx}. {title}\n\n")
        f.write(lyrics)
        f.write("\n\n---\n\n")

print(f"Saved {len(album.songs)} songs from '{ALBUM_NAME}' to {output_path}")
