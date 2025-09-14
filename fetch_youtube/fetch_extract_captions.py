import yt_dlp
import os
import re
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Env vars (replace hardcoded with these for security; set in terminal)
endpoint = "https://denni-meyguxi0-eastus2.cognitiveservices.azure.com/"
subscription_key = "124M9tbO3IjR7exQpd20rPvibS9wjjeYm8tXvWGnPaP43kfqxjisJQQJ99BHACHYHv6XJ3w3AAAAACOGtj4f"
api_version = "2024-12-01-preview"
deployment = "gpt-5-mini"  # Your deployment name

llm = AzureChatOpenAI(
    azure_deployment=deployment,
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

# Improved prompt for Chinese/Cantonese transcript
prompt_template = """
You are an expert at extracting recommendations for local Hong Kong restaurants from video transcripts (which may be in Chinese, Cantonese, or English). Focus exclusively on restaurants that are unique to Hong Kong or operated as independent/local establishments within HK—do not include any global or international chain restaurants (e.g., McDonald's, Starbucks, KFC, or similar franchises), even if mentioned. Only extract restaurants that are explicitly recommended or positively highlighted in the transcript as worth visiting.

For each qualifying restaurant, output the following fields:
- {{"Name"}}: The full name of the restaurant (in English if available, otherwise in Chinese with English transliteration if possible).
- {{"Location"}}: The specific address or neighborhood in Hong Kong (e.g., "Tsim Sha Tsui" or "123 Nathan Road, Kowloon").
- {{"Description/Highlights"}}: A brief summary of the restaurant's key features, cuisine, or signature dishes mentioned.
- {{"Why Recommended"}}: The reasons given in the transcript for why it's recommended (e.g., authentic flavors, value for money, unique experience).
- {{"Cost Estimate"}}: An approximate price range per person based on the transcript (e.g., "HKD 50-100" or "Budget-friendly").
- {{"Unique Tips"}}: Any special advice from the transcript, such as best time to visit, must-try items, or insider tips.

If no qualifying local HK restaurants are recommended, output an empty list. Transcript: {transcript}
Output strictly as a JSON list of objects (e.g., [{{"Name": "...", ...}}, ...]). Do not add any extra text or explanations.
"""

prompt = PromptTemplate.from_template(prompt_template)

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

def extract_fields(transcript):
    if not transcript:
        return []
    chain = prompt | llm | StrOutputParser()
    extracted = chain.invoke({"transcript": transcript})
    try:
        return json.loads(extracted)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON for transcript: {transcript[:100]}...")
        return []

def process_videos(video_ids):
    all_extracted = []
    for video_id in video_ids:
        print(f"Processing video: {video_id}")
        caption = download_and_clean_captions(video_id)
        if caption:
            fields = extract_fields(caption)
            all_extracted.extend(fields)
        else:
            print(f"No captions for video: {video_id}")
    return all_extracted

# List of video IDs (add yours here)
video_ids = [
    "zRR37i-SodU", 
    "-qR5lDgEB50",
    "tVuVADlJlA4",
    "wRw3DUOB6Ok",
    "g42wYfpYxPE",
    "ioRk3Cauu1U"
   
]

extracted_data = process_videos(video_ids)

# Save to JSON
with open('all_extracted_restaurants.json', 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, indent=4, ensure_ascii=False)

print("Extraction complete! Data saved to all_extracted_restaurants.json")