import pandas as pd
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))

df = pd.read_csv('data/brand_conversations_sample.csv')
print(f"Loaded {len(df)} conversations")

# Filter English only
def is_english(text):
    if pd.isna(text):
        return False
    text = str(text)
    try:
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / len(text) > 0.85 if len(text) > 0 else False
    except:
        return False

df['is_english'] = df['customer_message'].apply(is_english)
english_df = df[df['is_english']].copy()
print(f"English conversations: {len(english_df)} out of {len(df)}")

# Save english only
english_df.to_csv('data/brand_conversations_english.csv', index=False)
print("Saved English-only conversations to data/brand_conversations_english.csv")

# Sample 60 messages for intent discovery
sample_messages = english_df['customer_message'].dropna().sample(n=60, random_state=42).tolist()
messages_text = "\n".join([f"{i+1}. {str(msg)[:180]}" for i, msg in enumerate(sample_messages)])

prompt = f"""You are analyzing customer support messages sent to Amazon on Twitter.

Here are 60 real customer messages:

{messages_text}

Based on these messages, define exactly 7 intent categories that cover the main types of customer issues.
Make sure they are mutually exclusive and cover all cases.

For each intent return:
1. intent: short ALL_CAPS name
2. description: one sentence
3. examples: 3 example phrases

Return ONLY a JSON array, no other text:
[
  {{
    "intent": "ORDER_STATUS",
    "description": "Customer asking about order tracking, delivery status, or shipping updates",
    "examples": ["where is my order", "track my package", "when will it arrive"]
  }}
]"""

print("\nCalling Groq to define intents...")

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

intents_raw = response.choices[0].message.content.strip()
print("\nRaw response from Groq:")
print(intents_raw)

# Parse and save
try:
    # Remove markdown code blocks if present
    if '```' in intents_raw:
        parts = intents_raw.split('```')
        for part in parts:
            if part.startswith('json'):
                intents_raw = part[4:].strip()
                break
            elif '[' in part:
                intents_raw = part.strip()
                break

    # Find JSON array
    start = intents_raw.find('[')
    end = intents_raw.rfind(']') + 1
    if start != -1 and end != 0:
        intents_raw = intents_raw[start:end]

    intents = json.loads(intents_raw)

    with open('data/intents.json', 'w') as f:
        json.dump(intents, f, indent=2)

    print(f"\n✅ Saved {len(intents)} intents to data/intents.json")
    print("\n--- DEFINED INTENTS ---")
    for intent in intents:
        print(f"\n{intent['intent']}: {intent['description']}")
        print(f"  Examples: {', '.join(intent['examples'])}")

except json.JSONDecodeError as e:
    print(f"\nJSON parse error: {e}")
    print("Saving raw output...")
    with open('data/intents_raw.txt', 'w') as f:
        f.write(intents_raw)
    print("Saved to data/intents_raw.txt")
