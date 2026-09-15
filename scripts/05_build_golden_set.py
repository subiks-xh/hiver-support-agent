import pandas as pd
import os
import json
from dotenv import load_dotenv
from groq import Groq
import time

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = "openai/gpt-oss-120b"

with open('data/intents.json', 'r') as f:
    INTENTS = json.load(f)
INTENT_LIST = [i['intent'] for i in INTENTS]

df = pd.read_csv('data/brand_conversations_english.csv')
print(f"Loaded {len(df)} English conversations")

# Clean the dataframe
df = df.dropna(subset=['customer_message', 'brand_reply'])
df['customer_message'] = df['customer_message'].astype(str)
df['brand_reply'] = df['brand_reply'].astype(str)
df = df[df['customer_message'].str.len() > 5]
print(f"After cleaning: {len(df)} conversations")

# ============================================================
# SAMPLE STRATEGICALLY
# ============================================================
samples = []

# 1. Random sample (80 examples)
random_sample = df.sample(n=80, random_state=42)
for _, row in random_sample.iterrows():
    samples.append({
        'customer_message': str(row['customer_message']),
        'brand_reply': str(row['brand_reply']),
        'sampling_reason': 'random'
    })
print(f"Random stratum: {len(random_sample)}")

# 2. Short/vague messages under 100 chars (30 examples)
short_df = df[df['customer_message'].str.len() < 100]
short_count = min(30, len(short_df))
short_sample = short_df.sample(n=short_count, random_state=1)
for _, row in short_sample.iterrows():
    samples.append({
        'customer_message': str(row['customer_message']),
        'brand_reply': str(row['brand_reply']),
        'sampling_reason': 'short_message'
    })
print(f"Short message stratum: {short_count}")

# 3. Angry messages (30 examples) - broader keywords
angry_keywords = ['terrible|worst|horrible|ridiculous|unacceptable|disgrace|'
                  'pathetic|disgusting|outrageous|furious|angry|awful|useless|'
                  'incompetent|disappointed|shocking|appalling|rubbish|waste|'
                  'never again|so bad|very bad|extremely|!!!|wtf|omg']
angry_mask = df['customer_message'].str.lower().str.contains(
    angry_keywords[0], na=False, regex=True
)
angry_df = df[angry_mask]
angry_count = min(30, len(angry_df))
angry_sample = angry_df.sample(n=angry_count, random_state=2)
for _, row in angry_sample.iterrows():
    samples.append({
        'customer_message': str(row['customer_message']),
        'brand_reply': str(row['brand_reply']),
        'sampling_reason': 'angry_message'
    })
print(f"Angry message stratum: {angry_count}")

# 4. Messages with order/tracking numbers (30 examples)
order_mask = df['customer_message'].str.contains(r'\d{4,}', na=False, regex=True)
order_df = df[order_mask]
order_count = min(30, len(order_df))
order_sample = order_df.sample(n=order_count, random_state=3)
for _, row in order_sample.iterrows():
    samples.append({
        'customer_message': str(row['customer_message']),
        'brand_reply': str(row['brand_reply']),
        'sampling_reason': 'has_order_numbers'
    })
print(f"Order numbers stratum: {order_count}")

# 5. Question messages (20 examples)
question_mask = df['customer_message'].str.contains(r'\?', na=False, regex=True)
question_df = df[question_mask]
question_count = min(20, len(question_df))
question_sample = question_df.sample(n=question_count, random_state=4)
for _, row in question_sample.iterrows():
    samples.append({
        'customer_message': str(row['customer_message']),
        'brand_reply': str(row['brand_reply']),
        'sampling_reason': 'question_message'
    })
print(f"Question stratum: {question_count}")

# Deduplicate
seen = set()
unique_samples = []
for s in samples:
    key = str(s['customer_message'])[:80]
    if key not in seen:
        seen.add(key)
        unique_samples.append(s)

# Target 200 examples
unique_samples = unique_samples[:200]
print(f"\nTotal unique samples: {len(unique_samples)}")

from collections import Counter
reasons = Counter(s['sampling_reason'] for s in unique_samples)
print("Sampling breakdown:")
for reason, count in reasons.items():
    print(f"  {reason}: {count}")

# ============================================================
# AUTO-LABEL WITH LLM
# ============================================================
intent_schema = "\n".join([f"- {i['intent']}: {i['description']}" for i in INTENTS])

print(f"\nAuto-labeling {len(unique_samples)} examples with LLM...")
print("(This will take 4-6 minutes)\n")

labeled = []
errors = 0

for i, sample in enumerate(unique_samples):
    msg = str(sample['customer_message'])

    prompt = f"""Classify this Amazon customer support Twitter message into EXACTLY ONE intent.

Available intents (use ONLY these exact names):
{intent_schema}

Message: "{msg[:300]}"

Rules:
- You MUST pick one of the 7 intents listed above
- Never return "UNKNOWN" or any other value
- If unsure, pick the closest match

Also decide: ESCALATE or AUTO_HANDLE?
Escalate if: legal threats, fraud, hacking, very angry customer, payment dispute, or genuinely unclear.

Return JSON only (no extra text):
{{
  "intent": "ONE_OF_THE_7_INTENTS",
  "confidence": 0.95,
  "escalate": false,
  "escalation_reason": "brief reason here"
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()

        # Extract JSON
        start = raw.find('{')
        end = raw.rfind('}') + 1
        label = json.loads(raw[start:end])

        # Force valid intent
        detected_intent = label.get('intent', 'ORDER_STATUS')
        if detected_intent not in INTENT_LIST:
            detected_intent = 'ORDER_STATUS'

        labeled.append({
            'id': i,
            'customer_message': msg,
            'brand_reply': str(sample['brand_reply']),
            'sampling_reason': sample['sampling_reason'],
            'label_intent': detected_intent,
            'label_confidence': label.get('confidence', 0.5),
            'label_escalate': label.get('escalate', False),
            'label_escalation_reason': label.get('escalation_reason', ''),
            'human_verified': False,
            'human_intent': '',
            'human_escalate': '',
            'notes': ''
        })

        if (i + 1) % 25 == 0:
            print(f"   Labeled {i+1}/{len(unique_samples)}...")

        time.sleep(0.2)

    except Exception as e:
        errors += 1
        print(f"   Error on {i}: {str(e)[:80]}")
        labeled.append({
            'id': i,
            'customer_message': msg,
            'brand_reply': str(sample['brand_reply']),
            'sampling_reason': sample['sampling_reason'],
            'label_intent': 'ORDER_STATUS',
            'label_confidence': 0.3,
            'label_escalate': False,
            'label_escalation_reason': 'labeling error',
            'human_verified': False,
            'human_intent': '',
            'human_escalate': '',
            'notes': 'auto-label failed'
        })

golden_df = pd.DataFrame(labeled)
golden_df.to_csv('data/golden_set_raw.csv', index=False)

print(f"\n Saved {len(golden_df)} labeled examples to data/golden_set_raw.csv")
print(f"   Errors: {errors}")
print(f"\nIntent distribution:")
print(golden_df['label_intent'].value_counts())
print(f"\nEscalation distribution:")
print(golden_df['label_escalate'].value_counts())
print(f"\nSampling breakdown:")
print(golden_df['sampling_reason'].value_counts())
print("\n DONE! Now open data/golden_set_raw.csv and manually verify 50 rows")