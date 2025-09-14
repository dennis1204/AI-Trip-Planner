import yt_dlp
import os
import re
import json
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from googlesearch import search

# Env vars (use os.getenv for security; hardcoded here for demo)
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
You are an expert at extracting local restaurant recommendations from transcripts (in Chinese or English).
Extract all mentioned local restaurants from this transcript. For each, output fields: Name, Location, Description/Highlights, Why Recommended, Cost Estimate, Unique Tips.
Transcript: {transcript}
Output as JSON list.
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

def search_for_video_ids(query, num_results=20):
    video_ids = []
    for url in search(query, num_results=num_results):
        if "youtube.com/watch?v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
            video_ids.append(video_id)
    return video_ids[:10]  # Limit to 10 to avoid rate limits

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


# Example usage
query = "hk local restaurants recommended by locals or youtubers 2025 site:youtube.com"
video_ids = search_for_video_ids(query)
print(video_ids)
extracted_data = process_videos(video_ids)

# Automated search results (from web search; replace with dynamic if using a search library)
# Query: "hk local restaurants recommended by locals or youtubers 2025 site:youtube.com"
# video_ids_from_search = [
#     "HWMga1ULQEU",  # BEST HONG KONG FOOD!! 19 Meals
#     "oQQE3UQSORo",  # 25 MOST AFFORDABLE HONG KONG RESTAURANTS
#     "uaqPAnD-DUQ",  # Best Old School Trolley Dimsum in Hong Kong
#     "Cqbw-HtZyf8",  # Taking My Favourite Youtuber For His FIRST EVER DIM SUM
#     "4ehNo4zhwkQ",  # $100 Thailand Street Food Challenge (but HK related in search)
#     "jKXOh5wrzK4",  # DEEP Chinese Street Food Tour in Beijing (HK context in query)
#     "T8k2aECZSkA",  # Discover the Top 10 Most Popular Foods in Indonesia (HK in query)
#     "E2Rk3Cauu1fw",  # ISLAMIC CENTER CANTEEN YUM CHA MAKANAN MENU HALAL
#     "4Lkmnbeiits",  # Osaka Nightlife Guide (HK in query)
#     "yW1zlvwCdkE",  # Is This the Prettiest Place in Hawaii (HK in query)
#     "kcz-tQyHFA4",  # Ipoh Food Tour (HK in query)
#     "wbndEitaPH8",  # Ultimate Vancouver Food Guide (HK in query)
#     "cVoe_73O1NQ",  # TOUR OF HOTEL ELITEINN (HK in query)
#     "9anEeZpxuDs"  # Walk around infamous Kampung Air (HK in query)
# ] ['I-1IDOUHQ4A', 'PcdKZRAsOWo', 'HWMga1ULQEU', 'oQQE3UQSORo', 'BHigNaWjxEQ', '8vC1Wsc2_Tc', 'wUL-3GBJGao', 'Dsqox-l_am0', 'Bv9j7I2od8M', 'wnSDlmcvtUE']


# Process and extract
extracted_data = process_videos(video_ids)

# Save to JSON
with open('all_extracted_restaurants.json', 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, indent=4, ensure_ascii=False)

print("Extraction complete! Data saved to all_extracted_restaurants.json")