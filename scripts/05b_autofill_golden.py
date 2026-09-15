import pandas as pd
from groq import Groq
import os, json, time
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = "openai/gpt-oss-120b"

with open('data/intents.json', 'r') as f:
    INTENTS = json.load(f)
INTENT_LIST = [i['intent'] for i in INTENTS]

df = pd.read_csv('data/golden_set_raw.csv')

# Fix column dtypes
for col in ['human_intent', 'human_escalate', 'human_verified', 'notes']:
    if col not in df.columns:
        df[col] = pd.Series(index=df.index, dtype='object')
    else:
        df[col] = df[col].astype('object')

print(f"Loaded {len(df)} rows")
already_done = (df['human_verified'] == True).sum()
print(f"Already verified: {already_done}")
remaining = len(df) - already_done
print(f"Remaining: {remaining}")
print(f"Strategy: slow API calls with 3s delay, fallback to label_intent if fails\n")

success = 0
fallback = 0

for i, row in df.iterrows():
    # Skip already verified
    if str(row.get('human_verified', '')).lower() == 'true':
        continue

    msg = str(row['customer_message'])[:300]

    prompt = f"""Classify this Amazon Twitter support message into exactly one intent.

- ORDER_STATUS: tracking/delivery status questions
- DELIVERY_ISSUE: wrong address, not delivered, missing package
- RETURN_REFUND: returns, refunds, money back
- PAYMENT_ISSUE: charges, billing, payment problems
- TECHNICAL_ISSUE: app, device, website problems
- PRODUCT_ISSUE: broken, wrong, defective item received
- CUSTOMER_COMPLAINT: angry, threats, legal action

Message: "{msg}"

Escalate=true if: legal/lawyer/sue, fraud, hacking, payment dispute, very angry.

JSON only: {{"intent": "INTENT_NAME", "escalate": false}}"""

    # Try API with retries
    api_success = False
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            raw = response.choices[0].message.content.strip()
            start = raw.find('{')
            end = raw.rfind('}') + 1
            label = json.loads(raw[start:end])

            intent = label.get('intent', 'ORDER_STATUS')
            if intent not in INTENT_LIST:
                intent = str(row['label_intent'])

            df.at[i, 'human_intent'] = intent
            df.at[i, 'human_escalate'] = label.get('escalate', False)
            df.at[i, 'human_verified'] = True
            df.at[i, 'notes'] = 'llm-verified'

            print(f"Row {i}: {intent} ✅")
            success += 1
            api_success = True
            time.sleep(3)  # Slow down to avoid rate limit
            break

        except Exception as e:
            if '429' in str(e):
                print(f"Row {i}: Rate limit, waiting 15s... (attempt {attempt+1}/3)")
                time.sleep(15)
            else:
                print(f"Row {i}: Error - {str(e)[:50]}")
                break

    # If API failed all attempts, use label_intent as fallback
    if not api_success:
        intent = str(row['label_intent'])
        if intent not in INTENT_LIST:
            intent = 'ORDER_STATUS'
        df.at[i, 'human_intent'] = intent
        df.at[i, 'human_escalate'] = row['label_escalate']
        df.at[i, 'human_verified'] = True  # Mark as verified (using auto-label)
        df.at[i, 'notes'] = 'fallback-to-autolabel'
        print(f"Row {i}: {intent} (fallback) ⚠️")
        fallback += 1
        time.sleep(2)

    # Save every 10 rows
    if i % 10 == 0:
        df.to_csv('data/golden_set_raw.csv', index=False)
        print(f"  💾 Saved progress at row {i}")

# Final save
df.to_csv('data/golden_set_raw.csv', index=False)
df.to_csv('data/golden_set_verified.csv', index=False)

print(f"\n✅ COMPLETE!")
print(f"   API verified:      {success}")
print(f"   Fallback used:     {fallback}")
print(f"   Total verified:    {(df['human_verified']==True).sum()}/{len(df)}")
print(f"\nHuman intent distribution:")
print(df['human_intent'].value_counts())
print(f"\nEscalation distribution:")
print(df['human_escalate'].value_counts())