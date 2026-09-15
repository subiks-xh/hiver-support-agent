import pandas as pd
import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))
MODEL = "openai/gpt-oss-20b"

# Load intents
with open('data/intents.json', 'r') as f:
    INTENTS = json.load(f)

INTENT_LIST = [i['intent'] for i in INTENTS]

# Load conversation data for retrieval
print("Loading conversation data...")
conv_df = pd.read_csv('data/brand_conversations_english.csv')
print(f"Loaded {len(conv_df)} English conversations")

def get_intent_schema():
    lines = []
    for intent in INTENTS:
        lines.append(f"- {intent['intent']}: {intent['description']}")
    return "\n".join(lines)

# ============================================================
# STEP 1: CLASSIFY INTENT
# ============================================================
def classify_intent(customer_message: str) -> dict:
    intent_schema = get_intent_schema()

    prompt = f"""You are a customer support classifier for Amazon.

Classify the following customer message into exactly ONE of these intents:

{intent_schema}

Customer message: "{customer_message}"

Return a JSON object with exactly these fields:
- intent: the intent name (must be exactly one from the list above)
- confidence: float between 0.0 and 1.0
- reasoning: one sentence explaining why

Return ONLY the JSON object, no other text."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    raw = response.choices[0].message.content.strip()

    # Extract JSON
    try:
        start = raw.find('{')
        end = raw.rfind('}') + 1
        result = json.loads(raw[start:end])
    except:
        result = {"intent": "ORDER_STATUS", "confidence": 0.3, "reasoning": "Parse error"}

    # Validate intent
    if result.get('intent') not in INTENT_LIST:
        result['intent'] = 'ORDER_STATUS'
        result['confidence'] = 0.3

    return result

# ============================================================
# STEP 2: RETRIEVE SIMILAR EXAMPLES
# ============================================================
def retrieve_similar_examples(customer_message: str, n: int = 3) -> list:
    keywords = [w.lower() for w in str(customer_message).split() if len(w) > 3]

    filtered = conv_df.copy()
    filtered['score'] = 0

    for keyword in keywords[:6]:
        filtered['score'] += filtered['customer_message'].str.lower().str.contains(
            keyword, na=False, regex=False
        ).astype(int)

    top = filtered[filtered['score'] > 0].nlargest(n, 'score')

    examples = []
    for _, row in top.iterrows():
        examples.append({
            'customer': str(row['customer_message'])[:250],
            'brand_reply': str(row['brand_reply'])[:250],
            'score': row['score']
        })

    return examples

# ============================================================
# STEP 3: DRAFT REPLY
# ============================================================
def draft_reply(customer_message: str, intent: str, similar_examples: list) -> str:
    examples_text = ""
    if similar_examples:
        examples_text = "\n\nHere are similar past Amazon support conversations for reference:\n"
        for i, ex in enumerate(similar_examples, 1):
            examples_text += f"\nExample {i}:\nCustomer: {ex['customer']}\nAmazon replied: {ex['brand_reply']}\n"

    prompt = f"""You are an Amazon customer support agent responding on Twitter.

A customer sent this message (intent: {intent}):
"{customer_message}"
{examples_text}

Write a helpful, empathetic reply following these rules:
- Keep it under 280 characters when possible (Twitter limit)
- Be specific to their issue
- Match Amazon support tone: professional, apologetic for issues, solution-focused
- If you need more info, ask ONE clear question
- Do NOT invent order numbers, dates, or specific details
- Start with empathy if they are frustrated

Reply (just the reply text, nothing else):"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return response.choices[0].message.content.strip()

# ============================================================
# STEP 4: ESCALATION DECISION
# ============================================================
def decide_escalation(customer_message: str, intent: str, confidence: float) -> dict:

    # Hard rules first (fast, no API call needed)
    escalation_keywords = [
        'lawsuit', 'legal', 'lawyer', 'attorney', 'sue', 'court',
        'fraud', 'scam', 'stolen', 'hack', 'hacked', 'unauthorized',
        'police', 'fbi', 'crime', 'threat', 'kill', 'die',
        'media', 'news', 'press', 'journalist', 'viral',
        'discrimination', 'racist', 'abuse', 'harassment'
    ]

    message_lower = customer_message.lower()
    triggered = [k for k in escalation_keywords if k in message_lower]

    if triggered:
        return {
            'escalate': True,
            'action': 'ESCALATE',
            'reason': f"Sensitive keywords detected: {', '.join(triggered)}"
        }

    if confidence < 0.4:
        return {
            'escalate': True,
            'action': 'ESCALATE',
            'reason': f"Low classification confidence ({confidence:.2f}) - unclear intent needs human review"
        }

    if intent == 'PAYMENT_ISSUE':
        return {
            'escalate': True,
            'action': 'ESCALATE',
            'reason': "Payment/billing issues require human agent for security verification"
        }

    # For borderline cases, ask the LLM
    prompt = f"""A customer sent this message to Amazon support on Twitter:
"{customer_message}"

Intent classified as: {intent}

Should this be escalated to a human agent? Answer YES only if:
- The customer is extremely angry or threatening
- The issue is complex and needs account access
- A wrong auto-reply could cause significant harm
- It involves sensitive personal/financial data

Return JSON only:
{{"escalate": true or false, "reason": "one sentence max"}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = response.choices[0].message.content.strip()
        start = raw.find('{')
        end = raw.rfind('}') + 1
        decision = json.loads(raw[start:end])

        return {
            'escalate': decision.get('escalate', False),
            'action': 'ESCALATE' if decision.get('escalate') else 'AUTO_HANDLE',
            'reason': decision.get('reason', 'LLM decision')
        }
    except:
        return {
            'escalate': False,
            'action': 'AUTO_HANDLE',
            'reason': f"Standard {intent} - auto-handled"
        }

# ============================================================
# MAIN AGENT PIPELINE
# ============================================================
def run_agent(customer_message: str) -> dict:
    # Step 1: Classify
    classification = classify_intent(customer_message)
    intent = classification['intent']
    confidence = classification['confidence']

    # Step 2: Retrieve similar examples
    similar = retrieve_similar_examples(customer_message)

    # Step 3: Draft reply
    draft = draft_reply(customer_message, intent, similar)

    # Step 4: Escalation decision
    escalation = decide_escalation(customer_message, intent, confidence)

    return {
        'customer_message': customer_message,
        'intent': intent,
        'intent_confidence': confidence,
        'intent_reasoning': classification.get('reasoning', ''),
        'draft_reply': draft,
        'escalate': escalation['escalate'],
        'escalation_action': escalation['action'],
        'escalation_reason': escalation['reason'],
        'similar_examples_used': len(similar)
    }

# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    test_messages = [
        "Where is my order #114-5647382? It was supposed to arrive 3 days ago!",
        "I was charged twice for the same item. This is unacceptable!",
        "My package was delivered to the wrong address",
        "The product I received is completely broken and missing parts",
        "Amazon Video won't load on my Fire TV stick",
        "I want to sue Amazon for this terrible service, calling my lawyer tomorrow",
        "How do I return an item I bought last week?"
    ]

    print("=" * 60)
    print("TESTING AMAZON SUPPORT AGENT")
    print("=" * 60)

    for msg in test_messages:
        print(f"\n📨 CUSTOMER: {msg}")
        result = run_agent(msg)
        print(f"   🏷️  Intent: {result['intent']} (confidence: {result['intent_confidence']:.2f})")
        print(f"   ⚡ Action: {result['escalation_action']}")
        print(f"   📋 Reason: {result['escalation_reason']}")
        print(f"   💬 Draft:  {result['draft_reply'][:200]}")
        print("-" * 60)
