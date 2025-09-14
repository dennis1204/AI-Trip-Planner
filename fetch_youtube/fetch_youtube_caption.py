import yt_dlp
import os
import re

def download_and_clean_captions(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['zh-HK', 'en'],
        'skip_download': True,
        'outtmpl': 'captions.%(ext)s',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        caption_file = None
        for file in os.listdir('.'):
            if file.endswith('.vtt') or file.endswith('.srt'):
                caption_file = file
                break
        if caption_file:
            with open(caption_file, 'r', encoding='utf-8') as f:
                caption_text = f.read()
            # Clean and combine into paragraph
            lines = caption_text.strip().split('\n')
            caption_lines = [line for line in lines if not re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}$', line) and line.strip()]
            paragraph = ' '.join(caption_lines).replace('\n', ' ').strip()
            os.remove(caption_file)  # Clean up
            return paragraph
    return None

# Test
video_id = "tVuVADlJlA4"
paragraph_text = download_and_clean_captions(video_id)
if paragraph_text:
    print(paragraph_text)
else:
    print("No captions found.")