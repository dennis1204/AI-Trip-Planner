import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Azure OpenAI setup
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

# Simulated RAG-retrieved context (replace with actual RAG retrieval in full app)
rag_context = """
Event: Hong Kong Disneyland Halloween Time, October 2025, at Hong Kong Disneyland, Lantau Island. Description: Family-friendly Halloween events with parades and themed attractions. Ticket: ~$80 USD/adult.
Attraction: Hong Kong Disneyland, Lantau Island. Info: Theme park with rides, shows, and dining. Accessibility: Wheelchair-friendly, MTR to Disneyland Resort Station. Cost: ~$80 USD/adult, $60/child.
MTR Route: Tsuen Wan Line to Sunny Bay, transfer to Disneyland Resort Line. Fare: ~$3 USD/adult one-way.
Attraction: Temple Street Night Market, Yau Ma Tei. Info: Famous for street food (e.g., egg waffles, fish balls) and souvenirs. Cost: Free entry, food ~$5-10 USD/stall.
Bus Stop: Jordan Road, near Temple Street. Routes: KMB 7, 81. Fare: ~$1 USD.
Event: Hong Kong Food Festival, December 2025, at West Kowloon. Description: Showcases local and international cuisines. Cost: ~$20 USD entry.
"""

# System prompt for AI role
system_prompt = """
You are an expert HK travel planner. Use the provided data to create a personalized itinerary tailored to the user's requirements. Include day-by-day schedules with attractions/events, transport (MTR/bus), estimated costs in USD, and HK-specific tips (e.g., avoid crowds, use Octopus card). Ensure the total budget is respected. If data is missing, make reasonable assumptions based on typical HK tourism options.

Retrieved data:
{context}
""".format(context=rag_context)

# Chat history (starts with system prompt)
history = [
    {
        "role": "system",
        "content": system_prompt,
    }
]

# Continuous conversation loop
print("HK Travel Planner: Hi! Tell me your trip requirements (e.g., 'Plan a 3-day family trip to HK with $500 budget, focusing on Disneyland and food'). Type 'exit' to stop.")

while True:
    user_query = input("You: ").strip()
    if user_query.lower() == "exit":
        print("HK Travel Planner: Goodbye! Have a great trip.")
        break

    # Append user message to history
    history.append({"role": "user", "content": user_query})

    # Call Azure OpenAI with history
    response = client.chat.completions.create(
        messages=history,
        max_completion_tokens=16384,
        model=deployment
    )

    # Get AI response
    ai_response = response.choices[0].message.content
    print("HK Travel Planner:", ai_response)

    # Append AI response to history for memory
    history.append({"role": "assistant", "content": ai_response})