import os
from openai import AzureOpenAI

endpoint = "https://denni-meyguxi0-eastus2.cognitiveservices.azure.com/"
model_name = "gpt-5-mini"
deployment = "gpt-5-mini"

subscription_key = "124M9tbO3IjR7exQpd20rPvibS9wjjeYm8tXvWGnPaP43kfqxjisJQQJ99BHACHYHv6XJ3w3AAAAACOGtj4f"
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