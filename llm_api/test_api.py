import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


endpoint = os.getenv("AZURE_ENDPOINT")
subscription_key = os.getenv("AZURE_API_KEY")
model_name = "gpt-5-mini"
deployment = "gpt-5-mini"
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        }
    ],
    max_completion_tokens=16384,
    model=deployment
)

print(response.choices[0].message.content)