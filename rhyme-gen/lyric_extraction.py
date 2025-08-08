
import os
import re
import lyricsgenius
from dotenv import load_dotenv

load_dotenv()

GENIUS_API_TOKEN = os.getenv('GENIUS_API_TOKEN')

genius = lyricsgenius.Genius(
    GENIUS_API_TOKEN,
    timeout=15,
    retries=3
)

genius.skip_non_songs=True
genius.remove_section_headers=True

BASE_FOLDER = "albums"
os.makedirs(BASE_FOLDER, exist_ok=True)

ARTIST_NAME = "Kanye West"
ALBUM_NAME = "The College Dropout"

album = genius.search_album(ALBUM_NAME, ARTIST_NAME)

safe_album_folder = os.path.join(BASE_FOLDER, re.sub(r'[\\/*?:"<>| ]', "_", ALBUM_NAME))
os.makedirs(safe_album_folder, exist_ok=True)

for idx, track in enumerate(album.tracks, start=1):
    song = track.song
    title = song.title.strip()
    if 'skit' in title.lower():
        continue
    lyrics = song.lyrics.strip()

    safe_title = re.sub(r'[\\/*?:"<>| ]', "_", title)

    filename = f"{safe_title}.txt"
    filepath = os.path.join(safe_album_folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(lyrics)

    print(f"Wrote '{title}' -> {filepath}")

print(f"\nSaved {len(album.tracks)} tracks to folder '{safe_album_folder}/'")
